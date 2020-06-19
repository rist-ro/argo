import types

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from argo.core.network.AbstractGaussian import AbstractGaussian
from argo.core.utils.argo_utils import load_sonnet_module


class GaussianTriDiagonal(AbstractGaussian):

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
        mean, covariance, scale, L = self.create_mean_n_cov_layers(inputs)

        mean_t = mean
        covariance_t = covariance

        self.set_contractive_regularizer(mean, covariance,
                                         self._contractive_regularizer_inputs,
                                         self._contractive_regularizer_tuple,
                                         self._contractive_collection_network_str)

        # output_distribution = MultivariateNormalTriLChannelFlipped(loc=mean_t, scale_tril=L, validate_args=True)
        output_distribution = tfd.MultivariateNormalTriL(loc=mean_t, scale_tril=L, validate_args=True)

        # add reconstruction_node method (needed to some sort of mean or median to get reconstructions without sampling)
        def reconstruction_node(self):
            return self.mean()

        output_distribution.reconstruction_node = types.MethodType(reconstruction_node, output_distribution)

        self.mean = mean
        return output_distribution



    def mean(self):
        return self.mean

    def create_mean_n_cov_layers(self, inputs):
        # create the layers for mean and covariance

        module_name, module_kwargs = self._module_tuple
        extra_kwargs = self._extra_kwargs
        if module_name == "Linear" or module_name == "LinearWN":
            inputs = tf.layers.flatten(inputs)
            if self._output_shape is not None:
                self.check_key_not_in("output_size", module_kwargs)
                extra_kwargs["output_size"] = np.prod(self._output_shape)
            else:
                self._output_shape = [module_kwargs["output_size"]]

        elif module_name == "Conv2D" or module_name == "Conv2DWN":
            if self._output_shape is not None:
                self.check_key_not_in("output_channels", module_kwargs)
                self.check_key_not_in("stride", module_kwargs)
                self.check_key_not_in("padding", module_kwargs, 'SAME')

                extra_kwargs["output_channels"] = self._output_shape[2]
                extra_kwargs["stride"] = self._get_stride(inputs.shape.as_list()[1:],
                                                          self._output_shape)
        elif module_name == "Conv1D":
            if self._output_shape is not None:
                self.check_key_not_in("output_channels", module_kwargs)
                self.check_key_not_in("padding", module_kwargs, 'SAME')
                extra_kwargs["output_channels"] = self._output_shape[2]

        elif module_name == "Conv2DTranspose" or module_name == "Conv2DTransposeWN":
            if self._output_shape is not None:
                self.check_key_not_in("output_channels", module_kwargs)
                self.check_key_not_in("stride", module_kwargs)
                self.check_key_not_in("output_shape", module_kwargs)
                extra_kwargs["output_shape"] = self._output_shape[0:2]
                extra_kwargs["stride"] = self._get_stride(self._output_shape,
                                                          inputs.shape.as_list()[1:])
                extra_kwargs["output_channels"] = self._output_shape[2]


        else:
            raise Exception(
                "module name `%s` is not allowed, is not implemented yet to be wrapped by a Stochastic Layer." % module_name)

        kwargs = {**extra_kwargs,
                  **module_kwargs}

        sntmodule_mean = load_sonnet_module(module_name, kwargs)
        mean = sntmodule_mean(inputs)

        sntmodule_cov = load_sonnet_module(module_name, kwargs)
        covariance_params = sntmodule_cov(inputs)

        sntmodule_cov_plus = load_sonnet_module(module_name, kwargs)
        covariance_params_plus = sntmodule_cov_plus(inputs)

        # if I am using linear layers I need to reshape to match output_shape
        if module_name == "Linear" or module_name == "LinearWN":
            output_shape = [-1] + self._output_shape
            mean = tf.reshape(mean, output_shape)
            if not self._scalar_bool:
                covariance_params = tf.reshape(covariance_params, output_shape)

        covariance, L = self.get_covariance_from_parameters_tf(covariance_params, covariance_params_plus)
        # I don't need the standard deviation for the MultivariateFullCovariance
        scale = None
        mean = tf.transpose(mean, perm=[0, 2, 1])
        return mean, covariance, scale, L

    def get_covariance_from_parameters_tf(self, parameters, parameters_plus):

        unst_param = tf.unstack(parameters, axis=2)
        unst_param_plus = tf.unstack(parameters_plus, axis=2)

        covariances = []
        Ls = []

        for i in range(len(unst_param)):
            param = unst_param[i]
            param_plus = unst_param_plus[i]

            # loop through batches and tridiagonalize the covariance matrices
            # tf.nn.softplus(parameters)
            L1 = tf.map_fn(lambda x: diagonalize_covariance_matrix(x, epsilon=0), tf.nn.softplus(param))

            param_plus = tf.slice(param_plus, [0, 0], [-1, tf.shape(param_plus)[1] - 1])

            L2 = tf.map_fn(lambda x: diagonalize_covariance_matrix(x, epsilon=0), param_plus)
            L2 = tf.pad(L2, [[0, 0], [1, 0], [0, 1]], mode='CONSTANT')

            L = L1 + L2
            cov = tf.matmul(L, tf.transpose(L, perm=[0, 2, 1]))

            covariances += [cov]
            Ls += [L]

        covariance = tf.stack(covariances, axis=1)
        Lss = tf.stack(Ls, axis=1)

        return covariance, Lss


def tridiagonalize_covariance_matrix(params, params_plus):
    time_dim = tf.shape(params)[0]

    L1 = tf.zeros(shape=(time_dim, time_dim))
    diagonal1 = params
    L1 = tf.linalg.set_diag(L1, diagonal1)

    L2 = tf.zeros(shape=(time_dim - 1, time_dim - 1))
    diagonal2 = params_plus
    L2 = tf.linalg.set_diag(L2, diagonal2)
    L2 = tf.pad(L2, [[0, 1, 0], [0, 0, 1]], mode='CONSTANT')

    L = L1 + L2
    return tf.matmul(L, tf.transpose(L))


def diagonalize_covariance_matrix(params, epsilon = 0.01):
    time_dim = tf.shape(params)[0]

    L1 = tf.zeros(shape=(time_dim, time_dim))
    diagonal1 = params + epsilon
    L1 = tf.linalg.set_diag(L1, diagonal1)

    return L1
