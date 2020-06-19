import tensorflow as tf
# from tensorflow_probability import distributions as tfd

from argo.core.utils.argo_utils import tf_add_gaussian_noise_and_clip
from argo.core.network.Network import AbstractModule

# from argo.core.TFDeepLearningModel import load_network
# from argo.core.ArgoLauncher import ArgoLauncher
# from datasets.Dataset import Dataset
#
# import importlib
# import sys

class FromAE(AbstractModule):

    def __init__(self, filename, transform_prob=0.1, noisy_transform_prob=0., name='from_ae'):
        super().__init__(name=name)

        if not (0. <= transform_prob <= 1.):
            raise ValueError("`transform_prob` must be between 0 and 1, found `%f`" % transform_prob)

        self._transform_prob = transform_prob
        self._noisy_prob = tf.placeholder_with_default(noisy_transform_prob, shape=())
        self.random_number = tf.random_uniform((), 0, 1.0, dtype=tf.float32)

        # transform only a certain fraction of times
        self.t_cond = tf.less(self.random_number, self._transform_prob)

        self._frozen_graph_filename = filename

        # self._post_transform_noise = post_transform_noise
        # #IMPORT HACK DUE TO PACKAGE STRUCTURE...
        # sys.path.append("..")
        # dataset_params, model_params, config = ArgoLauncher.process_conf_file(conf_file)
        # f_name, m_name = model_params["model"].split(".")
        # module = importlib.import_module("vae.core."+f_name)
        # ModelClass = getattr(module, m_name)
        # # IMPORT HACK DUE TO PACKAGE STRUCTURE...
        #
        # with self._enter_variable_scope():
        #     self.random_number = tf.random_uniform((), 0, 1.0, dtype=tf.float32)
        #     # self.t_cond = tf.less(self.random_number, self._transform_prob)
        #     self._ae_network, self._checkpoint_name = load_constructed_network(ModelClass, conf_file, global_step=global_step)
        #
        # # # IMPORT HACK DUE TO PACKAGE STRUCTURE...
        # sys.path.remove("..")
        # # # IMPORT HACK DUE TO PACKAGE STRUCTURE...

    def _transform(self, inputs):
        scope = "import"
        # x_shape = (None,)+dataset.x_shape
        # x = tf.placeholder(tf.float32, shape=x_shape)
        with tf.gfile.GFile(self._frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, input_map={"inputs": inputs, "noisy_prob": self._noisy_prob}, name=scope)

        full_scope = self.scope_name + "/" + scope
        transformed = graph.get_tensor_by_name(full_scope + "/transformed:0")
        # make sure shapes are the same or compatible with inputs
        transformed.set_shape(inputs.shape)

        # if self._post_transform_noise>0:
        #     transformed, noise = tf_add_gaussian_noise_and_clip(transformed,
        #                                                         self._post_transform_noise,
        #                                                         low=0,
        #                                                         high=1)

        return transformed

    # def _transform(self, inputs):
    #     encoder_module = self._ae_network.encoder_module
    #     decoder_module = self._ae_network.decoder_module
    #
    #     net = encoder_module(inputs)
    #     if isinstance(net, tfd.Distribution):
    #         net = net.sample()
    #
    #     net = decoder_module(net)
    #     if isinstance(net, tfd.Distribution):
    #         net = net.reconstruction_node()
    #
    #     return net

    def _build(self, inputs):
        transformed_inputs = self._transform(inputs)

        # transform only a certain fraction of times
        # t_cond decides to transform the whole batch ATM (for efficiency)
        # other option for image by image preprocessing, woud be to include
        # the transformation in the dataset augmentation with `map`.
        # In case this seems needed, ask me (Riccardo)
        outputs = tf.cond(self.t_cond,
                         lambda: transformed_inputs,
                         lambda: inputs
                         )

        return outputs

# def _var_dict(self):
#     scope_name = self.scope_name+"/"
#     # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#     # print_tensors_in_checkpoint_file(self._checkpoint_name, all_tensors=False, tensor_name='', all_tensor_names=True)
#     var_name_dict = {v.op.name[len(scope_name):] : v for v in self._ae_network.get_variables()}
#     return var_name_dict

# def load_constructed_network(ModelClass, conffile, global_step=None):
#     dataset_params, model_params, config = ArgoLauncher.process_conf_file(conffile)
#     dataset = Dataset.load_dataset(dataset_params)
#     x_shape = (None,) + dataset.x_shape
#     x = tf.placeholder(tf.float32, shape=x_shape, name='dummy_input')
#     network, checkpoint_name = load_network(ModelClass, conffile, global_step=global_step)
#     stuffs = network(x)
#     return network, checkpoint_name


