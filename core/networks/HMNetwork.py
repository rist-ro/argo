import numpy as np
import tensorflow as tf
from sonnet import BatchFlatten

from argo.core.network.Bernoulli import Bernoulli
from argo.core.network.BernoulliPlusMinusOne import BernoulliPlusMinusOne
from argo.core.network.ArgoNetworkWithDefaults import ArgoNetworkWithDefaults
from core.networks.GenerationNetwork import GenerationNetwork
from core.networks.RecognitionNetwork import RecognitionNetwork


class HMNetwork(ArgoNetworkWithDefaults):
    """
    Network class for managing a VAE network
    """

    # if not present in self._opts, the values specified in default_params will be provided automagically
    # see more in AbstractNetwork.__init__
    default_params = {
        "network_architecture": {
            "layers": [10, 10]
        },
        "samples":              10,
        "pm_one":               True,
    }

    def create_id(self):
        layers_ids = "_".join(map(str, self._layers))

        _id = '-l_' + layers_ids

        super_id = super().create_id()

        _id += super_id
        return _id

    def __init__(self, opts, clip_probs=0, pm=True, name="hm_network"):
        """Short summary.

        Args:
            self._opts (dict): parameters of the task.
            name (str): name of the Sonnet module.
        """

        super().__init__(opts, name, None)

        self._network_architecture = self._opts["network_architecture"]
        self._layers = self._network_architecture["layers"]
        self._clip_probs = clip_probs
        self._pm = pm

        self._regs = self.get_regularizers(self._default_weights_reg, self._default_bias_reg)

    def _build(self, x, b_size, n_z_samples):
        """
        Args:
            x (tf.tensor): input node.
        """

        input_shape = x.shape.as_list()[1:]

        x = BatchFlatten()(x)
        x = tf.tile(x, [n_z_samples, 1])

        rec_mod = RecognitionNetwork(self._layers,
                                     clip_probs=self._clip_probs,
                                     pm=self._pm,
                                     initializers={
                                         "w": self._default_weights_init,
                                         "b": self._default_bias_init},
                                     regularizers=self.get_regularizers(self._default_weights_reg,
                                                                        self._default_bias_reg))
        gen_mod = GenerationNetwork(self._layers,
                                    np.prod(input_shape),
                                    pm=self._pm,
                                    initializers={
                                        "w": self._default_weights_init,
                                        "b": self._default_bias_init},
                                    regularizers=self.get_regularizers(self._default_weights_reg,
                                                                       self._default_bias_reg),
                                    clip_probs=self._clip_probs)

        mega_batch_size = b_size * n_z_samples
        size_prior = tf.placeholder_with_default(mega_batch_size, shape=None)

        if self._pm:
            h_prior_plus_minus_one = BernoulliPlusMinusOne(output_size=self._layers[-1],
                                                           initializers={
                                                               "w": tf.constant_initializer(0.0),
                                                               "b": self._default_bias_init},
                                                           regularizers=self.get_regularizers(
                                                               bias_reg=self._default_bias_reg),
                                                           clip_value=self._clip_probs,
                                                           dtype=x.dtype,
                                                           name="top_bias")(tf.zeros([size_prior, 1]))
        else:
            h_prior_plus_minus_one = Bernoulli(output_size=self._layers[-1],
                                               initializers={
                                                   "w": tf.constant_initializer(0.0),
                                                   "b": self._default_bias_init},
                                               regularizers=self.get_regularizers(
                                                   bias_reg=self._default_bias_reg),
                                               clip_value=self._clip_probs,
                                               dtype=x.dtype,
                                               name="top_bias")(tf.zeros([size_prior, 1]))

        h_prior_sample = h_prior_plus_minus_one.sample()
        with tf.name_scope("wake"):
            h_inferred, self._hrw = rec_mod(x)

            x_reconstruct, x_reconstruct_distr, self._hgw = gen_mod(h_prior_sample, h_distr=h_prior_plus_minus_one,
                                                                    hr=self._hrw)

            x_reconstruct = tf.reshape(x_reconstruct, [-1] + input_shape)

        with tf.name_scope("sleep"):
            h_prior_sample2 = h_prior_plus_minus_one.sample()

            x_inferred, x_inferred_distr, self._hgs = gen_mod(h_prior_sample2, h_distr=h_prior_plus_minus_one)

            h_reconstruct, self._hrs = rec_mod(x_inferred, hg=self._hgs)

            x_inferred = tf.reshape(x_inferred, [-1] + input_shape)

        return {
            "wake":  (h_inferred, x_reconstruct, x_reconstruct_distr, self._hrw, self._hgw),
            "sleep": (h_reconstruct, x_inferred, x_inferred_distr, self._hrs, self._hgs),
            "prior": (h_prior_plus_minus_one, h_prior_sample2)
        }

    def get_regularizers(self, weights_reg=None, bias_reg=None):
        regs = {}
        if weights_reg:
            regs["w"] = weights_reg
        if bias_reg:
            regs["b"] = bias_reg
        return regs
