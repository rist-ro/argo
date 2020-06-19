import tensorflow as tf

from .AbstractModule import AbstractModule


class Tanh(AbstractModule):
    
    def __init__(self, name='Tanh'):

        super().__init__(name = name)
                
    def _build(self, inputs):

        return tf.nn.tanh(inputs)
