import tensorflow as tf
import sonnet as snt
import numpy as np

from operator import xor

import types

import pdb

from abc import ABC, abstractmethod

from tensorflow_probability import distributions as tfd

from .AbstractModule import AbstractModule


class OneHotCategorical(AbstractModule):
    """
    Multinomial distribution for one hot data
    """

    def __init__(self, output_size=-1, output_shape=-1, initializers={}, regularizers={}, clip_value=0, dtype=None,
                 name='OneHotCategorical'):

        super().__init__(name=name)

        assert xor(output_size == -1,
                   output_shape == -1), "Either output_size or output_shape mut be specified, not both"

        if output_size != -1:
            self._output_shape = [output_size]
        else:
            self._output_shape = output_shape

        self._initializers = initializers
        self._regularizers = regularizers

        self._clip_value = clip_value
        self._dtype = dtype

    def _build(self, inputs):
        # create the layers for the probs of the distribution

        output_shape = [-1] + self._output_shape
        logits = tf.reshape(
            snt.Linear(np.prod(self._output_shape), initializers=self._initializers, regularizers=self._regularizers)(
                inputs), output_shape)

        dtype = inputs.dtype
        if self._dtype is not None:
            dtype = self._dtype

        # we need to ensure that the probs sum up to 1, so we pass the logits through a softmax
        probs = tf.contrib.layers.softmax(logits, scope=None)
        probs = tf.clip_by_value(probs, self._clip_value, 1 - self._clip_value)
        onehot_categorical = tf.contrib.distributions.OneHotCategorical(probs=probs, dtype=dtype)

        def reconstruction_node(self):
            return self.probs

        onehot_categorical.reconstruction_node = types.MethodType(reconstruction_node, onehot_categorical)

        def distribution_parameters(self):
            return [self.probs]

        onehot_categorical.distribution_parameters = types.MethodType(distribution_parameters, onehot_categorical)

        return onehot_categorical
