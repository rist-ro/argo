import types

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from core.wavenet.MultiVariateNormalChannelFlipped import MultiVariateNormalChannelFlipped

from argo.core.network.GaussianDiagonal import GaussianDiagonal as GaussianD


class GaussianDiagonal(GaussianD):

    def __init__(self, module_tuple=("Linear", {}),

                 output_size=None,
                 output_shape=None,
                 minimal_covariance=0.,
                 covariance_parameterization="softplus",
                 scalar_covariance=False,
                 initializers={},
                 regularizers={},
                 contractive_regularizer=None,
                 name='gaussian_diagonal'):
        super().__init__(module_tuple=module_tuple,
                         output_size=output_size,
                         output_shape=output_shape,
                         minimal_covariance=minimal_covariance,
                         covariance_parameterization=covariance_parameterization,
                         scalar_covariance=scalar_covariance,
                         initializers=initializers,
                         regularizers=regularizers,
                         contractive_regularizer=contractive_regularizer,
                         name=name)

    def _build(self, inputs):
        mean, covariance, scale = self.create_mean_n_cov_layers(inputs)

        mean = tf.transpose(mean, perm=[0, 2, 1])
        scale = tf.transpose(scale, perm=[0, 2, 1])

        self.set_contractive_regularizer(mean, covariance,
                                         self._contractive_regularizer_inputs,
                                         self._contractive_regularizer_tuple,
                                         self._contractive_collection_network_str)

        output_distribution = tfd.MultivariateNormalDiag(loc=mean, scale_diag=scale)

        # add reconstruction_node method (needed to some sort of mean or median to get reconstructions without sampling)
        def reconstruction_node(self):
            return self.mean()

        output_distribution.reconstruction_node = types.MethodType(reconstruction_node, output_distribution)

        return output_distribution
