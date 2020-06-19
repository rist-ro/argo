import numpy as np
import tensorflow as tf
from tensorflow.python.layers.base import _add_elements_to_collection

from core.HM import PHASE_WAKE, PHASE_SLEEP
from core.optimizers.WakeSleepOptimizer import WakeSleepOptimizer


class WakeSleepGradientDescentOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, **optimizer_kwargs):
        self._model = optimizer_kwargs["model"]

        self._individual_learning_rate = optimizer_kwargs["individual_learning_rate"]

        self._learning_rate = optimizer_kwargs["learning_rate"]
        self._rescale_learning_rate = optimizer_kwargs["rescale_learning_rate"]

        self._d_p = None

        self._grads_w = []
        self._grads_s = []
        super().__init__(self._learning_rate)

    def compute_gradients(self, phase, loss, *args, **kw):

        hgw = self._model._hgw
        hrw = self._model._hrw
        hrs = self._model._hrs
        hgs = self._model._hgs

        weights = []
        grads_and_vars = []
        if phase == PHASE_WAKE:
            for i in range(len(hgw))[::-1]:
                distr_p = hgw[i][0]
                sample_q = tf.stop_gradient(hrw[i][1])

                distr_q = hrs[i][0]
                sample_p = tf.stop_gradient(hgs[i][1])

                if i == len(hgw) - 1:
                    current_weights = [WakeSleepOptimizer._get_bias()]  # Size of the last hidden layer
                else:
                    current_weights = [*WakeSleepOptimizer._get_layer_vars("gen_net", "l_g_{}".format(i))]

                likelihood = self.get_loss(distr_p, sample_q, distr_q, sample_p, phase)

                grad = tf.gradients(likelihood, current_weights)
                grads_and_vars += [(g, v) for (g, v) in zip(grad, current_weights)]

                weights += current_weights
            self._grads_w = grads_and_vars
        elif phase == PHASE_SLEEP:
            for i in range(len(hrs) - 1):
                distr_q = hrs[i + 1][0]
                sample_p = tf.stop_gradient(hgs[i + 1][1])

                distr_p = hgw[i+1][0]
                sample_q = tf.stop_gradient(hrw[i+1][1])

                sample_p = tf.stop_gradient(sample_p)
                likelihood = self.get_loss(distr_p, sample_q, distr_q, sample_p, phase)

                current_weights = [*WakeSleepOptimizer._get_layer_vars("rec_net", "l_r_{}".format(i))]

                grad = tf.gradients(likelihood, current_weights)

                grads_and_vars += [(g, v) for (g, v) in zip(grad, current_weights)]

                weights += current_weights

            self._grads_s = grads_and_vars
        else:
            raise ValueError("invalid value for phase '{}'".format(phase))

        lr = 1.
        if phase == PHASE_SLEEP:
            lr *= self._rescale_learning_rate

        grads_and_vars_not_none = [(lr * g, v) for (g, v) in grads_and_vars if g is not None]

        assert np.all([g.shape == v.shape for (g, v) in
                       grads_and_vars_not_none]), "The shapes of weights and gradients are not the same"

        return grads_and_vars_not_none

    def get_loss(self, distr_p, sample_q, distr_q, sample_p, phase):
        if phase == PHASE_SLEEP:
            likelihood_per_node = -distr_q.log_prob(sample_p)
        else:
            likelihood_per_node = -distr_p.log_prob(sample_q)
        # sum over the layer
        likelihood_per_layer = tf.reduce_sum(likelihood_per_node, axis=-1)
        # average over the batch
        likelihood = tf.reduce_mean(likelihood_per_layer, axis=0)
        return likelihood
