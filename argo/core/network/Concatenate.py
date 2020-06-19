import tensorflow as tf

from .AbstractModule import AbstractModule

import pdb

class Concatenate(AbstractModule):
    
    def __init__(self, node_name, channel_wise=False, name='Concatenate'):

        super().__init__(name=name)

        self._node_name = node_name
        self._channel_wise = channel_wise
        
    def _build(self, inputs):

        node = tf.get_default_graph().get_tensor_by_name(self._node_name + ":0")

        if self._channel_wise:
            img_size = inputs.get_shape()[1]
            node_image_like = tf.keras.layers.Dense(img_size * img_size)(node)
            node_to_be_concatenated = tf.reshape(node_image_like, [-1, img_size, img_size, 1])
        else:
            n_replicate = tf.shape(inputs)[0]/tf.shape(node)[0]
            shape_tile = [1]*len(node.get_shape())
            shape_tile[0] = n_replicate
            node_to_be_concatenated = tf.tile(node, shape_tile)

        return tf.concat([inputs, node_to_be_concatenated], axis=-1)
