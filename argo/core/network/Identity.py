import tensorflow as tf

from .AbstractModule import AbstractModule

import pdb

class Identity(AbstractModule):
    
    def __init__(self, name='Identity'):

        self._name = name
        
        super().__init__(name = name)
        
    def _build(self, inputs):
        
        return tf.identity(inputs)
