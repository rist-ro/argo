import tensorflow as tf

from .AbstractModule import AbstractModule


class Sigmoid(AbstractModule):
    
    def __init__(self, name='Sigmoid'):

        super().__init__(name = name)
                
    def _build(self, inputs):

        return tf.nn.sigmoid(inputs)
