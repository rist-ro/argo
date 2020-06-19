import numpy as np
import tensorflow as tf

from core.HM import PHASE_WAKE, PHASE_SLEEP
from core.optimizers.WakeSleepOptimizer import WakeSleepOptimizer


class ReweightedWakeSleepOptimizer(WakeSleepOptimizer):

    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)

        self._b_size = self._model.b_size
        self._n_samples = self._model.n_z_samples

        self._sleep_balance = 0.5
        self._wake_q = 1.0 - self._sleep_balance
        self._sleep_q = self._sleep_balance

        self._qbaseline = tf.constant(0.)
        if optimizer_kwargs["q_baseline"]:
            self._qbaseline = tf.constant(1.) / tf.cast(self._n_samples, dtype=tf.float32)

    def compute_gradients(self, phase, k=1, *args, **kw):
        weights = []
        grads = []

        if phase == PHASE_WAKE:
            hrw = self._model._hrw
            hgw = self._model._hgw
            self._individual_learning_rate = self.check_ilr(self._individual_learning_rate, hrw)

            weights = [self._get_bias()]  # Size of the last hidden layer

            difference_b = hrw[-1][1] - hgw[-1][0].mean()

            # CALC WEIGHT PART
            imp_weights_p = self.compute_normalized_weights(hr=hrw, hg=hgw)
            imp_weights_q = imp_weights_p - self._qbaseline
            # END OF CALC WEIGHT PART

            difference_b = tf.einsum("b,bv->bv", imp_weights_p, difference_b)
            difference_b = tf.reduce_sum(
                tf.reshape(difference_b, [self._n_samples, self._b_size, difference_b.shape.as_list()[1]]),
                axis=0)
            grads = [
                tf.multiply(self._individual_learning_rate[-1], tf.reduce_mean(difference_b,
                                                                               axis=0),
                            name="Wb{}".format(len(hrw) - 1))]

            for i in range(len(hrw) - 1)[::-1]:
                weights += [*self._get_layer_vars("gen_net", "l_g_{}".format(i))]

                difference_b = hrw[i][1] - hgw[i][0].mean()
                difference_w = tf.einsum("bu,bv->bvu", difference_b, hrw[i + 1][1])

                # CALC WEIGHT PART
                difference_b = tf.einsum("b,bv->bv", imp_weights_p, difference_b)
                difference_b = tf.reduce_sum(
                    tf.reshape(difference_b, [self._n_samples, self._b_size, difference_b.shape.as_list()[1]]), axis=0)
                difference_w = tf.einsum("b,buv->buv", imp_weights_p, difference_w)
                difference_w = tf.reshape(difference_w,[self._n_samples, self._b_size, *difference_w.shape.as_list()[1:3]])
                difference_w = tf.reduce_sum(difference_w, axis=0)
                # END OF CALC WEIGHT PART

                grads += [tf.multiply(self._individual_learning_rate[i], tf.reduce_mean(difference_w, axis=0),
                                      name="Ww{}".format(i)),
                          tf.multiply(self._individual_learning_rate[i], tf.reduce_mean(difference_b, axis=0),
                                      name="Wb{}".format(i))]

            # WAKE PHASE SLEEP
            for i in range(len(hrw) - 1):
                weights += [*WakeSleepOptimizer._get_layer_vars("rec_net", "l_r_{}".format(i))]

                difference_b = hrw[i + 1][1] - hrw[i + 1][0].mean()
                difference_w = tf.einsum("bu,bv->bvu", difference_b, hrw[i][1])

                # CALC WEIGHT PART
                difference_b = tf.einsum("b,bv->bv", imp_weights_q, difference_b)
                difference_b = tf.reduce_sum(
                    tf.reshape(difference_b, [self._n_samples, self._b_size, difference_b.shape.as_list()[1]]), axis=0)
                difference_w = tf.einsum("b,buv->buv", imp_weights_q, difference_w)
                difference_w = tf.reshape(difference_w,
                                          [self._n_samples, self._b_size, *difference_w.shape.as_list()[1:3]])
                difference_w = tf.reduce_sum(difference_w, axis=0)
                # END OF CALC WEIGHT PART

                grads += [
                    tf.multiply(self._wake_q * self._individual_learning_rate[i], tf.reduce_mean(difference_w, axis=0),
                                name="WSw{}".format(i)),
                    tf.multiply(self._wake_q * self._individual_learning_rate[i], tf.reduce_mean(difference_b, axis=0),
                                name="WSb{}".format(i))]

        elif phase == PHASE_SLEEP:
            # CLASSIC SLEEP
            hrs = self._model._hrs
            hgs = self._model._hgs

            self._individual_learning_rate = self.check_ilr(self._individual_learning_rate, hrs)

            for i in range(len(hrs) - 1):
                weights += [*self._get_layer_vars("rec_net", "l_r_{}".format(i))]

                difference_b = hgs[i + 1][1] - hrs[i + 1][0].mean()
                difference_w = tf.einsum("bu,bv->bvu", difference_b, hgs[i][1])

                # CALC WEIGHT PART - NO WEIGHTS =1
                difference_b = tf.reduce_mean(
                    tf.reshape(difference_b, [self._n_samples, self._b_size, difference_b.shape.as_list()[1]]), axis=0)
                difference_w = tf.reshape(difference_w,
                                          [self._n_samples, self._b_size, *difference_w.shape.as_list()[1:3]])
                difference_w = tf.reduce_mean(difference_w, axis=0)
                # END OF CALC WEIGHT PART

                grads += [
                    tf.multiply(self._sleep_q * self._individual_learning_rate[i], tf.reduce_mean(difference_w, axis=0),
                                name="Sw{}".format(i)),
                    tf.multiply(self._sleep_q * self._individual_learning_rate[i], tf.reduce_mean(difference_b, axis=0),
                                name="Sb{}".format(i))]

        else:
            raise ValueError("invalid value for phase '{}'".format(phase))

        lr = 1.
        if phase == PHASE_SLEEP:
            lr *= self._rescale_learning_rate

        regs = self.get_regularizers(weights)

        grads_and_vars_not_none = [(tf.multiply(-lr, g, name="g_" + g.name.split(":")[0]) + r, v) for (g, r, v) in
                                   zip(grads, regs, weights) if g is not None]

        assert np.all([g.shape == v.shape for (g, v) in
                       grads_and_vars_not_none]), "The shapes of weights and gradients are not the same"

        return grads_and_vars_not_none

    def compute_normalized_weights(self, hr, hg):

        with tf.name_scope("reweights"):
            log_probs_p = 0.0
            log_probs_q = 0.0

            for i in range(len(hg)):
                samples_q = hr[i][1]

                distr_p = hg[i][0]
                log_probs_p_all = tf.reduce_sum(distr_p.log_prob(samples_q), axis=-1)
                log_probs_p += log_probs_p_all

                distr_q = hr[i][0]
                if i > 0:
                    log_probs_q_all = tf.reduce_sum(distr_q.log_prob(samples_q), axis=-1)
                else:
                    log_probs_q_all = tf.zeros(tf.shape(samples_q)[0])

                log_probs_q += log_probs_q_all

            unnormalized_weight_log = self.get_unnormalized_weigth_log(log_probs_p, log_probs_q)

            unnormalized_weight_log_reduced = tf.reduce_logsumexp(
                tf.reshape(unnormalized_weight_log, [self._n_samples, self._b_size]), axis=0)

            normalized_weights_log = unnormalized_weight_log - tf.tile(unnormalized_weight_log_reduced,
                                                                       [self._n_samples])
            normalized_weights = tf.exp(normalized_weights_log)
        return normalized_weights

    def get_unnormalized_weigth_log(self, log_probs_p, log_probs_q):
        return log_probs_p - log_probs_q
