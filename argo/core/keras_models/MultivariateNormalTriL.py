import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from .keras_utils import MINIMAL_COVARIANCE
from .ArgoKerasModel import ArgoKerasModel

class MultivariateNormalTriL(ArgoKerasModel):

    def _id(self):
        _id = 'TriL'
        _id += "_b"+str(int(self._bl))
        _id += "_fl" + str(int(self._fl))
        return _id

    def __init__(self, output_size, bayesian_layers=False, flipout=True, layer_kwargs = {}, layer_kwargs_bayes = {}):
        super().__init__(name='TriL')

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
        self.dense_loc = Dense(output_size)
        self.dense_diag_params = Dense(output_size)
        n_out_of_diag_elems = int(output_size * (output_size-1)/2)
        self.dense_out_of_diag_params = Dense(n_out_of_diag_elems)

        n_tril = n_out_of_diag_elems + output_size
        self._calibration_tril_params = self.add_weight("calibration_tril_params",
                                                       shape=(n_tril,),
                                                       trainable=False,
                                                       initializer=tf.initializers.constant(value=1.))
        self.calibration_tril = tf.contrib.distributions.fill_triangular(self._calibration_tril_params, name="calibration_tril")

    def call(self, inputs, training=False, **extra_kwargs):
        inputs = self._flat(inputs)
        loc = self.dense_loc(inputs)
        diag_params = self.dense_diag_params(inputs)
        out_of_diag_params = self.dense_out_of_diag_params(inputs)

        lower_triangle = tf.contrib.distributions.fill_triangular(out_of_diag_params)
        lower_triangle = tf.pad(lower_triangle, [[0, 0], [1, 0], [0, 1]])

        diag_positive = MINIMAL_COVARIANCE + tf.nn.softplus(diag_params)

        scale_tril = tf.linalg.set_diag(lower_triangle, diag_positive)

        ouput_params = {"loc" : loc, "scale_tril" : tf.multiply(self.calibration_tril, scale_tril)}

        distr = tfp.distributions.MultivariateNormalTriL(**ouput_params)

        # hack because keras does not want distr in output... (Riccardo)
        distr.shape = tf.TensorShape(tuple(distr.batch_shape.as_list() + distr.event_shape.as_list()))

        return distr
