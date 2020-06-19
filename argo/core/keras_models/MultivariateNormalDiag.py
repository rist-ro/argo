import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial

from .keras_utils import MINIMAL_COVARIANCE
from .ArgoKerasModel import ArgoKerasModel

class MultivariateNormalDiag(ArgoKerasModel):
    
    def _id(self):
        _id = 'D'
        _id += "_b"+str(int(self._bl))
        _id += "_fl" + str(int(self._fl))
        return _id

    def __init__(self, output_size, bayesian_layers=False, flipout=False, layer_kwargs = {}, layer_kwargs_bayes = {}):
        super().__init__(name='norm_diag')

        self._bl = bayesian_layers
        self._fl = flipout

        if bayesian_layers:
            if flipout:
                Dense = partial(tfp.layers.DenseFlipout, **layer_kwargs_bayes)
            else:
                Dense = partial(tfp.layers.DenseReparameterization, **layer_kwargs_bayes)
        else:
            Dense = partial(tf.keras.layers.Dense, **layer_kwargs)

        self._flat = tf.keras.layers.Flatten()
        self._output_size = output_size
        self.dense_loc = Dense(output_size)
        self.dense_diag_params = Dense(output_size)

    def call(self, inputs, training=False, **extra_kwargs):
        inputs = self._flat(inputs)
        loc = self.dense_loc(inputs)
        diag_params = self.dense_diag_params(inputs)
        scale_diag = MINIMAL_COVARIANCE + tf.nn.softplus(diag_params)

        ouput_params = {"loc": loc, "scale_diag": scale_diag}
        distr = tfp.distributions.MultivariateNormalDiag(**ouput_params)

        # hack because keras does not want distr in output... (Riccardo)
        distr.shape = tf.TensorShape(tuple(distr.batch_shape.as_list() + distr.event_shape.as_list()))

        return distr

    # def compute_output_shape(self, input_shape):
    #     output_shape = [input_shape[0]] + [self._output_size]
    #     return tuple(output_shape)
