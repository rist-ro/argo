import tensorflow as tf
import sonnet as snt

import pdb

from .AbstractModule import AbstractModule


class RandomUniform(AbstractModule):

    # TODO move to *kwargs?
    def __init__(self, shape, minval, maxval, seed=None, rate=None, name="RandomUniform"):

        super().__init__(name = name)

        self._shape = shape
        self._minval = minval
        self._maxval = maxval
        self._seed = seed
    
    def _build(self, input):
        # input is ignored
        return tf.random_uniform(self._shape,
                                 minval = self._minval,
                                 maxval = self._maxval,
                                 seed = self._seed,
                                 name = self.module_name,
        )
    
