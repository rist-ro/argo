import tensorflow as tf
import sonnet as snt

import pdb

from .AbstractModule import AbstractModule


class AveragePooling2D(AbstractModule):

    def __init__(self, pool_size, strides, padding="valid", name="AveragePooling2D"):

        super().__init__(name = name)
        self._pool_size = pool_size
        self._strides = strides
        self._padding = padding
        self._name = name

    def _build(self, input):
        return tf.layers.average_pooling2d(input,
                                       pool_size = self._pool_size,
                                       strides = self._strides,
                                       padding = self._padding,
                                       name = self._name
        )
