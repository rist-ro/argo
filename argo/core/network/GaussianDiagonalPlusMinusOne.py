import tensorflow as tf
import sonnet as snt

import numpy as np

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from .AbstractGaussian import AbstractGaussian

import types

class GaussianDiagonalPlusMinusOne(AbstractGaussian):
    
    def __init__(self, module_tuple = ("Linear", {}),
                 output_size=None,
                 output_shape=None,
                 minimal_covariance=0,
                 covariance_parameterization="softplus",
                 scalar_covariance = False,
                 initializers = {},
                 regularizers = {},
                 contractive_regularizer = None,
                 name='gaussian_diagonal_zero_one'):

        super().__init__(module_tuple = module_tuple,
                         output_size = output_size,
                         output_shape = output_shape,
                         minimal_covariance = minimal_covariance,
                         covariance_parameterization=covariance_parameterization,
                         scalar_covariance = scalar_covariance,
                         initializers = initializers,
                         regularizers = regularizers,
                         contractive_regularizer = contractive_regularizer,
                         name = name)

    def _build(self, inputs):
        mean, covariance, scale = self.create_mean_n_cov_layers(inputs)
        
        mean_plus_minus_one = tf.tanh(mean)
        
        self.set_contractive_regularizer(mean_plus_minus_one, covariance,
                                        self._contractive_regularizer_inputs,
                                        self._contractive_regularizer_tuple,
                                        self._contractive_collection_network_str)
        
        output_distribution = tfd.Normal(loc=mean_plus_minus_one, scale=scale)
        
        # add reconstruction_node method (needed to some sort of mean or median to get reconstructions without sampling)
        def reconstruction_node(self):
            return self.mean()
        
        output_distribution.reconstruction_node = types.MethodType(reconstruction_node, output_distribution)
        
        def distribution_parameters(self):
            return [mean_plus_minus_one, np.square(scale)]
        output_distribution.distribution_parameters = types.MethodType(distribution_parameters, output_distribution)

        return output_distribution
    
