import tensorflow as tf

from argo.core.network.Bernoulli import Bernoulli
from argo.core.network.BernoulliPlusMinusOne import BernoulliPlusMinusOne
from argo.core.network.AbstractModule import AbstractModule


class GenerationNetwork(AbstractModule):

    def __init__(self, layers, final_size, clip_probs, pm, initializers={}, regularizers={}, name="gen_net"):
        super().__init__(name=name)
        self._layers = layers
        self._final_size = final_size
        self._initializers = initializers
        self._regularizers = regularizers
        self._clip_probs = clip_probs
        self._pm = pm
        self.reverse_layers = [self._final_size] + [*self._layers[:-1]]

    def _build(self, h, h_distr=None, hr=None):
        """
        Args:
            h (tf.tensor): input node.
        """

        gen_layer = h
        self._hg = [(h_distr, gen_layer)]
        distr = None
        for i, layer_size in list(enumerate(self.reverse_layers))[::-1]:
            print("GENNET: creating layer {} with layersize: {}".format(i, layer_size))

            if self._pm:
                layer = BernoulliPlusMinusOne(output_size=layer_size, initializers=self._initializers,
                                              regularizers=self._regularizers,
                                              clip_value=self._clip_probs, dtype=h.dtype, name="l_g_{}".format(i))
            else:
                layer = Bernoulli(output_size=layer_size, initializers=self._initializers,
                                  regularizers=self._regularizers,
                                  clip_value=self._clip_probs, dtype=h.dtype, name="l_g_{}".format(i))

            if hr is not None:
                distr = layer(tf.stop_gradient(hr[i + 1][1]))
            else:
                distr = layer(tf.stop_gradient(gen_layer))

            gen_layer = distr.sample()

            self._hg.append((distr, gen_layer))

        x_reconstruct = gen_layer
        x_reconstruct_distr = distr

        return x_reconstruct, x_reconstruct_distr, self._hg[::-1]
