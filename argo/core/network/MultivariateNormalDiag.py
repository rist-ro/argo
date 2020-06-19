import tensorflow as tf
from tensorflow_probability import distributions as tfd
from functools import partial
from .AbstractGaussianSimple import AbstractGaussianSimple
import types
import sonnet as snt

class MultivariateNormalDiag(AbstractGaussianSimple):

    def __init__(self,
                 output_size,
                 minimal_covariance=0.,
                 initializers={},
                 regularizers={},
                 custom_getter={},
                 name='normal_diag'):
        super().__init__(output_size=output_size,
                         minimal_covariance=minimal_covariance,
                         initializers=initializers,
                         regularizers=regularizers,
                         custom_getter=custom_getter,
                         name=name)


    def _build(self, inputs):

        inputs = tf.layers.flatten(inputs)

        self.dense_loc = snt.Linear(self._output_size, **self._extra_kwargs)
        self.dense_diag_params = snt.Linear(self._output_size, **self._extra_kwargs)

        loc = self.dense_loc(inputs)
        diag_params = self.dense_diag_params(inputs)
        scale_diag = self._minimal_covariance + tf.nn.softplus(diag_params)

        ouput_params = {"loc": loc, "scale_diag": scale_diag}
        distr = tfd.MultivariateNormalDiag(**ouput_params)

        return distr

