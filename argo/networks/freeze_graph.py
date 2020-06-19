import pbtxt
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
import tensorflow as tf
import pdb
import os
import urllib.request
import tarfile
os.environ["CUDA_VISIBLE_DEVICES"] ="2"

'''
bazel-bin/tensorflow/python/tools/freeze_graph 
--input_graph=some_graph_def.pb \
--input_checkpoint=model.ckpt-8361242 \
--output_graph=/tmp/frozen_graph.pb --output_node_names=softmax

input_graph=/home/scopeserver/RaidDisk/DeepLearning/mwang/Model_backup2/slim/graph.pbtxt 
--input_checkpoint=/home/scopeserver/RaidDisk/DeepLearning/mwang/Model_backup2/slim/inception_resnet_v2_2016_08_30.ckp
t
 --output_graph=/tmp/frozen_graph.pb 
--output_node_names=InceptionResnetV2/Logits/Predictions
'''
tf.app.flags.DEFINE_string(
    'input_graph', 'inception_v4.pbtxt', 'The graph structure to freeze pbtxt file')
tf.app.flags.DEFINE_string(
    'input_saver', '', 'TensorFlow saver file to load')
# tf.app.flags.DEFINE_string(
  #  'input_checkpoint', 'inception_v4.ckpt', 'variables to load in the graph')
tf.app.flags.DEFINE_integer(
    'checkpoint_version', 2,
    'file format')
tf.app.flags.DEFINE_string(
    'output_graph', 'inception_v4.pb', 'filename should end with .pb')
tf.app.flags.DEFINE_string(
    'output_node_names', '', 'The name of the output nodes')
tf.app.flags.DEFINE_string(
    'initializer_nodes', '', 'initializer nodes to run before testing')
tf.app.flags.DEFINE_boolean(
    'clear_devices', True,
    'Whether to remove device specifications.')
# FLAGS.input_binary
tf.app.flags.DEFINE_boolean(
    'input_binary', True,
    '?')

tf.app.flags.DEFINE_string(
    'variable_names_whitelist', '', 'variables to convert to constants')
tf.app.flags.DEFINE_string(
    'variable_names_blacklist', '', 'variables to skip from converting to constants')
tf.app.flags.DEFINE_string(
    'input_meta_graph', '', 'meta graph')
tf.app.flags.DEFINE_string(
    'input_saved_model_dir', '', 'Path to the dir with TensorFlow \'SavedModel\' file and variables')
tf.app.flags.DEFINE_string(
    'saved_model_tags', 'serve', 'Group of tag(s) of the MetaGraphDef to load, in string format,\
      separated by \',\'. For tag-set contains multiple tags, all tags \
      must be passed in.\
      ')
tf.app.flags.DEFINE_string(
    'restore_op_name', 'save/restore_all', 'The name of the master restore operator')
tf.app.flags.DEFINE_string(
    'filename_tensor_name', 'save/Const:0', 'The name of the tensor holding the save path')

FLAGS = tf.app.flags.FLAGS

url='http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz'
filename = url.split('/')[-1]

current_directory = os.getcwd()
main_directory=os.path.join(current_directory,filename)
outputFileName='inception_v4.ckpt'
outputFilePath=os.path.join(current_directory,outputFileName)

if not os.path.exists(outputFilePath):

    main_directory, _ = urllib.request.urlretrieve(url=url, filename=main_directory)
    file=tarfile.open(name=main_directory,mode="r:gz")
    #pdb.set_trace()
    file.extract(outputFileName,path=current_directory)
    file.close()
   # pdb.set_trace()
    os.remove(main_directory)

def main(_):
    
    if not FLAGS.output_node_names:
        raise ValueError('You must supply the output nodes from which the activation needs to be calculated --output_node_names')
   
    if FLAGS.checkpoint_version == 1:
        checkpoint_version = saver_pb2.SaverDef.V1
    elif FLAGS.checkpoint_version == 2:
        checkpoint_version = saver_pb2.SaverDef.V2
    else:
        raise ValueError("Invalid checkpoint version (must be '1' or '2'): %d" %
                     flags.checkpoint_version)
   # pdb.set_trace()
    freeze_graph.freeze_graph(
        input_graph=FLAGS.input_graph,
        input_saver= FLAGS.input_saver,
        input_binary=FLAGS.input_binary,
        output_node_names=FLAGS.output_node_names,
        restore_op_name=FLAGS.restore_op_name,
        filename_tensor_name=FLAGS.filename_tensor_name,
        output_graph=FLAGS.output_graph,
        clear_devices=FLAGS.clear_devices,
        initializer_nodes=FLAGS.initializer_nodes,
        variable_names_whitelist=FLAGS.variable_names_whitelist,
        variable_names_blacklist=FLAGS.variable_names_blacklist,
        input_meta_graph=FLAGS.input_meta_graph,
        input_saved_model_dir=FLAGS.input_saved_model_dir,
        saved_model_tags=FLAGS.saved_model_tags,
        checkpoint_version=checkpoint_version,
        input_checkpoint=outputFileName)

if __name__ == '__main__':
  tf.app.run()
