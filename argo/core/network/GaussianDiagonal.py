import tensorflow as tf
import sonnet as snt

import numpy as np

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from .AbstractGaussian import AbstractGaussian

import pdb

import types

class GaussianDiagonal(AbstractGaussian):
    
    def __init__(self, module_tuple = ("Linear", {}),
                 output_size=None,
                 output_shape=None,
                 minimal_covariance=0.,
                 covariance_parameterization="softplus",
                 scalar_covariance = False,
                 initializers = {},
                 regularizers = {},
                 contractive_regularizer = None,
                 name='gaussian_diagonal'):

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

        self.set_contractive_regularizer(mean, covariance,
                                         self._contractive_regularizer_inputs,
                                         self._contractive_regularizer_tuple,
                                         self._contractive_collection_network_str)

      
        output_distribution = tfd.Normal(loc=mean, scale=scale)
        
        # add reconstruction_node method (needed to some sort of mean or median to get reconstructions without sampling)
        def reconstruction_node(self):
            return self.mean()
        
        output_distribution.reconstruction_node = types.MethodType(reconstruction_node, output_distribution)
        
        def params(self):
            return (self.mean(), tf.square(self.stddev()))
        
        output_distribution.params = types.MethodType(params, output_distribution)

        def default_prior(self, prior_shape):
            zeros = tf.zeros(shape=prior_shape)
            ones = tf.ones(shape=prior_shape)
            prior = tfd.Normal(loc=zeros, scale=ones, name="prior")
            return prior

        output_distribution.default_prior = types.MethodType(default_prior, output_distribution)
        
        return output_distribution
