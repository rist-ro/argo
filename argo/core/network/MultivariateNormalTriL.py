import tensorflow as tf
from tensorflow_probability import distributions as tfd
from functools import partial
from .AbstractGaussianSimple import AbstractGaussianSimple
import types
import sonnet as snt

class MultivariateNormalTriL(AbstractGaussianSimple):

    def __init__(self,
                 output_size,
                 minimal_covariance=0.,
                 initializers={},
                 regularizers={},
                 custom_getter={},
                 name='normal_tril'):
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
        n_out_of_diag_elems = int(self._output_size * (self._output_size - 1) / 2)
        self.dense_out_of_diag_params = snt.Linear(n_out_of_diag_elems, **self._extra_kwargs)


        loc = self.dense_loc(inputs)
        diag_params = self.dense_diag_params(inputs)
        out_of_diag_params = self.dense_out_of_diag_params(inputs)

        lower_triangle = tf.contrib.distributions.fill_triangular(out_of_diag_params)
        lower_triangle = tf.pad(lower_triangle, [[0, 0], [1, 0], [0, 1]])

        diag_positive = self._minimal_covariance + tf.nn.softplus(diag_params)

        scale_tril = tf.linalg.set_diag(lower_triangle, diag_positive)

        dtype = inputs.dtype
        n_tril = n_out_of_diag_elems + self._output_size
        self._calibration_tril_params = tf.get_variable("calibration_tril_params",
                                                        shape=(n_tril,),
                                                        dtype=dtype,
                                                        trainable=False,
                                                        initializer=tf.initializers.constant(value=1.))

        self.calibration_tril = tf.contrib.distributions.fill_triangular(self._calibration_tril_params, name="calibration_tril")


        ouput_params = {"loc" : loc, "scale_tril" : tf.multiply(self.calibration_tril, scale_tril)}

        distr = tfd.MultivariateNormalTriL(**ouput_params)

        return distr

