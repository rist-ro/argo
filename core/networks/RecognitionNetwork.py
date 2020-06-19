import tensorflow as tf

from argo.core.network.Bernoulli import Bernoulli
from argo.core.network.BernoulliPlusMinusOne import BernoulliPlusMinusOne
from argo.core.network.AbstractModule import AbstractModule


class RecognitionNetwork(AbstractModule):

    def __init__(self, layers, clip_probs, pm, initializers={}, regularizers={}, name="rec_net"):
        super(RecognitionNetwork, self).__init__(name=name)
        self._layers = layers
        self._initializers = initializers
        self._regularizers = regularizers
        self._clip_probs = clip_probs
        self._pm = pm

    def _build(self, x, hg=None):
        """
        Args:
            x (tf.tensor): input node.

        """
        rec_layer = x

        self._hr = [(None, rec_layer)]
        for i, layer_size in enumerate(self._layers):
            print("RECNET: creating layer {} with layersize: {}".format(i,layer_size))

            if self._pm:
                layer = BernoulliPlusMinusOne(output_size=layer_size, initializers=self._initializers, regularizers=self._regularizers, clip_value=self._clip_probs,
                                  dtype=x.dtype, name="l_r_{}".format(i))
            else:
                layer = Bernoulli(output_size=layer_size, initializers=self._initializers, regularizers=self._regularizers, clip_value=self._clip_probs,
                                              dtype=x.dtype, name="l_r_{}".format(i))

            if hg is not None:
                distr = layer(tf.stop_gradient(hg[i][1]))
            else:
                distr = layer(tf.stop_gradient(rec_layer))

            rec_layer = distr.sample()

            self._hr.append((distr, rec_layer))

        h = rec_layer
        return h, self._hr

