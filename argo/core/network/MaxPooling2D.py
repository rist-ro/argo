import tensorflow as tf
import sonnet as snt

import pdb

from .AbstractModule import AbstractModule


class MaxPooling2D(AbstractModule):

    def __init__(self, pool_size, strides, padding="valid", data_format="channels_last", name="MaxPooling2D"):

        super().__init__(name = name)
                
        self._pool_size = pool_size
        self._strides = strides
        self._padding = padding
        self._data_format = data_format
    
    def _build(self, input):
        return tf.layers.max_pooling2d(input,
                                       pool_size = self._pool_size,
                                       strides = self._strides,
                                       padding = self._padding,
                                       data_format = self._data_format,
                                       name = self.module_name
        )
    
