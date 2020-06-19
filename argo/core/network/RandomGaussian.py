import tensorflow as tf
import sonnet as snt

import pdb

from .AbstractModule import AbstractModule


class RandomGaussian(AbstractModule):

    # TODO move to *kwargs?
    def __init__(self, shape, seed=None, rate=None, name="RandomUniform"):

        super().__init__(name = name)

        self._shape = shape
        self._seed = seed
    
    def _build(self, input):
        # input is ignored
        return tf.random_normal(self._shape,
                                mean=0.0,
                                stddev=1.0,
                                seed = self._seed,
                                name = self.module_name,
        )
    
