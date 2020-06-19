import tensorflow as tf
from tensorflow import keras

class FromCallable(keras.layers.Layer):
    def __init__(self, callable, kwargs):
        super().__init__()
        self._cf = callable
        self._cf_kwargs = kwargs

    def call(self, inputs):
        return self._cf(inputs, **self._cf_kwargs)

