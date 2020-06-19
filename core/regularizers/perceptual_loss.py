import tensorflow as tf

from argo.core.utils.argo_utils import CUSTOM_REGULARIZATION

# from tensorflow.core.framework import graph_pb2
# from tensorflow.python.platform import gfile

from tensorflow.python.framework import importer


# see https://stackoverflow.com/questions/56324534/tensorflow-delete-nodes-from-graph
def delete_ops_from_graph(graph_def, op_to_be_deleted):
    #with open(input_model_filepath, 'rb') as f:
    #    graph_def = tf.GraphDef()
    #    graph_def.ParseFromString(f.read())

    # Delete nodes
    nodes = []
    for node in graph_def.node:
        if op_to_be_deleted in node.name:
            print('Drop', node.name)
        else:
            nodes.append(node)

    mod_graph_def = tf.GraphDef()
    mod_graph_def.node.extend(nodes)

    # Delete references to deleted nodes
    for node in mod_graph_def.node:
        inp_names = []
        for inp in node.input:
            if op_to_be_deleted in inp:
                #pdb.set_trace()
                #inp_names.append('Placeholder')
                pass
            else:
                inp_names.append(inp)

        # delete the expected input type also
        # del node.attr[:]
        del node.input[:]
        node.input.extend(inp_names)

    #with open(output_model_filepath, 'wb') as f:
    #    f.write(mod_graph_def.SerializeToString())

    return mod_graph_def
  
def replicate(param, n_z_samples):
    num_shapes = len(param.shape.as_list()[1:])
    ones = [1] * num_shapes
    param_replicate = tf.tile(param, [n_z_samples] + ones)
    return param_replicate


# vae specific regularizers
# see https://medium.com/miccai-educational-initiative/tutorial-abdominal-ct-image-synthesis-with-variational-autoencoders-using-pytorch-933c29bb1c90
def perceptual_loss(model, pb, input, scale, nodes=[], matching=[]):

    #pb = 'core/argo/networks/inception_v4.pb'
    #pb = '/home/luigi/prediction/natural/MNIST-c-st0/FF-cCE-st0-stp0-bs32-trGD_lr0.01-cNo-nD200_D200_D10-cpS-aR-wix-bic0.1-r0/saved_models/frozen_graph.pb'
    #x = tf.image.resize(model.x, [28,28])

    x = model.x
    rec = model.x_reconstruction_node

    # load graph
    # TF 1.14 graph_def = tf.contrib.gan.eval.get_graph_def_from_disk(pb)
    # since TF 1.15 moved to tensorflow-gan, thus I manually load
    # with gfile.FastGFile(pb, 'rb') as f:
    #      graph_def = graph_pb2.GraphDef.FromString(f.read())

    with tf.gfile.GFile(pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    matched_nodes = []
    for match in matching:
        matched_nodes += [n.name+":0" for n in graph_def.node if match in n.name]

    print("found matching nodes: `{:}`".format(matched_nodes))

    # # fix batch norm nodes old bug not needed tf1.15
    # for node in graph_def.node:
    #     if node.op == 'RefSwitch':
    #         node.op = 'Switch'
    #         for index in xrange(len(node.input)):
    #             if 'moving_' in node.input[index]:
    #                 node.input[index] = node.input[index] + '/read'
    #     elif node.op == 'AssignSub':
    #         node.op = 'Sub'
    #         if 'use_locking' in node.attr: del node.attr['use_locking']
    #

    output_tensor = nodes + matched_nodes #"ff_network/network/Relu_2:0"
    #output_tensor = "InceptionV4/Logits/AvgPool_1a/AvgPool:0"
    #output_tensor = "ff_network/network/features:0"

    def print_nodes(graph_def, nodes):
        for node in graph_def.node:
            print(node.name)
            if node.name in nodes:
                print(node)

    #graph_def = delete_ops_from_graph(graph_def, "IteratorGetNext")

    # see run_image_classifier in https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
    def run_feature_extractor(tensor, graph_def, input_tensor, output_tensor, scope="feature_extractor"):

        input_map = {input_tensor: tensor}
        
        is_singleton = isinstance(output_tensor, str)
        if is_singleton:
            output_tensor = [output_tensor]

        # tf.import_graph_def(graph_def, input_map=input_map, name=scope)
        #
        # graph = tf.get_default_graph()
        #
        # feature = graph.get_tensor_by_name(scope + "/feature_name:0")

        features = importer.import_graph_def(graph_def, input_map, output_tensor, name=scope)

        if is_singleton:
            features = features[0]

        return features

    #pdb.set_trace()

    #with tf.variable_scope('perceptual_loss'):
    #input = 'Placeholder:0'

    features_extractor_network = lambda x: run_feature_extractor(x, graph_def, input, output_tensor)
    
    input_shape = x.shape.as_list()[1:]
    ones = [1] * len(input_shape)
    x_replicate = tf.tile(x, [model.n_z_samples] + ones)

    features_x = features_extractor_network(x_replicate)
    features_rec = features_extractor_network(rec)

    # check the norm is correct
    l1_norms = [tf.norm(features_x[i] - features_rec[i], ord=1) for i in range(len(output_tensor))]

    regularizer = tf.add_n(l1_norms)
    scaled_regularizer = tf.multiply(scale, regularizer)

    tf.add_to_collection(CUSTOM_REGULARIZATION, scaled_regularizer)

    return scaled_regularizer, [regularizer], "perceptual_loss"
