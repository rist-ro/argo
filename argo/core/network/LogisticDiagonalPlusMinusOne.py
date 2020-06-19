
from tensorflow_probability import distributions as tfd
# from tensorflow_probability import bijectors as tfb

from .AbstractLogistic import AbstractLogistic

import types

class LogisticDiagonalPlusMinusOne(AbstractLogistic):
    
    def __init__(self, module_tuple = ("Linear", {}),
                 output_size=None,
                 output_shape=None,
                 minimal_covariance=0,
                 covariance_parameterization="softplus",
                 scalar_covariance = False,
                 initializers = {},
                 regularizers = {},
                 contractive_regularizer = None,
                 plus_minus_one_method = "tanh",
                 name='logistic_diagonal_plus_minus_one'):

        super().__init__(module_tuple = module_tuple,
                         output_size = output_size,
                         output_shape = output_shape,
                         minimal_covariance = minimal_covariance,
                         covariance_parameterization = covariance_parameterization,
                         scalar_covariance = scalar_covariance,
                         initializers = initializers,
                         regularizers = regularizers,
                         contractive_regularizer = contractive_regularizer,
                         name = name)
        
        self.plus_minus_one_method = plus_minus_one_method
    
    def _build(self, inputs):
        
        mean, covariance, scale = self.create_mean_n_cov_layers(inputs)
        
        mean_plus_minus_one = self.force_between_plus_minus_one(mean, self.plus_minus_one_method)
        
        # TODO if you want contractive regularizers implement them first. Then, uncomment the following lines (Riccardo)
        # self.set_contractive_regularizer(mean_zero_one, covariance,
        #                                 self._contractive_regularizer_inputs,
        #                                 self._contractive_regularizer_tuple,
        #                                 self._contractive_collection_network_str)
        #
        
        output_distribution = tfd.Logistic(loc=mean_plus_minus_one, scale=scale)
        
        # add reconstruction_node method (needed to some sort of mean or median to get reconstructions without sampling)
        def reconstruction_node(self):
            return self.mean()
        
        output_distribution.reconstruction_node = types.MethodType(reconstruction_node, output_distribution)
        
        return output_distribution
    
