from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf
import imagenet
from tensorflow.python.platform import gfile
from inception_v4 import inception_v4
from inception_v4 import inception_v4_arg_scope
from inception_v4 import inception_v4_base
import pdb
import functools
slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

# tf.app.flags.DEFINE_string('dataset_name', 'imagenet',
#                            'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'output_file', 'inception_v4.pbtxt', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './', 'Directory to save intermediate dataset files to')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')



tf.app.flags.DEFINE_bool('write_text_graphdef', False,
                         'Whether to write a text version of graphdef.')

FLAGS = tf.app.flags.FLAGS

# inception_network ={
#                 'inception_v1': inception.inception_v1,
#                 'inception_v2': inception.inception_v2,
#                 'inception_v3': inception.inception_v3,
#                 'inception_v4': inception.inception_v4,
#                 'inception_resnet_v2': inception.inception_resnet_v2,

#                 }

# inception_arg_scopes_map={
#                   'inception_v1': inception.inception_v3_arg_scope,
#                   'inception_v2': inception.inception_v3_arg_scope,
#                   'inception_v3': inception.inception_v3_arg_scope,
#                   'inception_v4': inception.inception_v4_arg_scope,
#                   'inception_resnet_v2': inception.inception_resnet_v2_arg_scope,
#                   }

def inception_v4_get_network_fn( num_classes, weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.
OA
  Args:
    num_classes: The number of classes to use for classification. If 0 or None,
      the logits layer is omitted and its input features are returned instead.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
          net, end_points = network_fn(images)
      The `images` input is a tensor of shape [batch_size, height, width, 3]
      with height = width = network_fn.default_image_size. (The permissibility
      and treatment of other sizes depends on the network_fn.)
      The returned `end_points` are a dictionary of intermediate activations.
      The returned `net` is the topmost layer, depending on `num_classes`:
      If `num_classes` was a non-zero integer, `net` is a logits tensor
      of shape [batch_size, num_classes].
      If `num_classes` was 0 or `None`, `net` is a tensor with the input
      to the logits layer of shape [batch_size, 1, 1, num_features] or
      [batch_size, num_features]. Dropout has not been applied to this
      (even if the network's original classification does); it remains for
      the caller to do this or not.

  Raises:
    ValueError: If network `name` is not recognized.
  """
#  if name not in inception_networks:
 #   raise ValueError('Name of network unknown %s' % name)
  func = inception_v4
 # pdb.set_trace()
  @functools.wraps(func)
  def network_fn(images, **kwargs):
    arg_scope =inception_v4_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes=num_classes, is_training=is_training,
                  **kwargs)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn

def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    #dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
     #                                       FLAGS.dataset_dir)
    image_net=imagenet.get_split('train', FLAGS.dataset_dir ,file_pattern=None, reader= None)
    inception_network_fn = inception_v4_get_network_fn(
        num_classes=(image_net.num_classes - FLAGS.labels_offset),
        is_training=False)
    #pdb.set_trace()
    image_size = inception_network_fn.default_image_size
    input_shape = [FLAGS.batch_size, image_size, image_size, 3]
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=input_shape)
    inception_network_fn(placeholder)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    graph_def = graph.as_graph_def()
    if FLAGS.write_text_graphdef:
      tf.io.write_graph(
          graph_def,
          os.path.dirname(FLAGS.output_file),
          os.path.basename(FLAGS.output_file),
          as_text=True)
    else:
      with gfile.GFile(FLAGS.output_file, 'wb') as f:
        f.write(graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run()
