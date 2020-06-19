import tensorflow as tf
import sonnet as snt

import numpy as np

from tensorflow_probability import distributions as tfd
#from tensorflow_probability import bijectors as tfb

import sonnet as snt

from .AbstractGaussian import AbstractGaussian

import types

class Gaussian(AbstractGaussian):
    
    def __init__(self, module_tuple = ("Linear", {}),

                 output_size=None,
                 output_shape=None,
                 minimal_covariance=0.,

                 covariance_parameterization="softplus",
                 scalar_covariance = False,
                 initializers = {},
                 regularizers = {},
                 contractive_regularizer = None,
                 name='gaussian'):
        
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
        # TODO this could be adapter to be compatible with non linear layers
        #mean, covariance, scale = self.create_mean_n_cov_layers(inputs)
        inputs = tf.layers.flatten(inputs)
        
        linear_mean = snt.Linear(output_size=output_size)
        mean = linear_mean(inputs)
        pdb.set_trace()
        linear_cov = snt.Linear(output_size=[output_size, output_size])
        cov = linear_mean(inputs)
        
        # TODO this neds to be adapter to the new covariance
        #self.set_contractive_regularizer(mean, covariance,
        #                                self._contractive_regularizer_inputs,
        #                                self._contractive_regularizer_tuple,
        #                                self._contractive_collection_network_str)
        
        output_distribution = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
        
    # add reconstruction_node method (needed to some sort of mean or median to get reconstructions without sampling)
    def reconstruction_node(self):
        return self.mean()
        
        output_distribution.reconstruction_node = types.MethodType(reconstruction_node, output_distribution)    
        return output_distribution

