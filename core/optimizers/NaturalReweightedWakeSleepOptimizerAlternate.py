import numpy as np
import tensorflow as tf

from core.HM import PHASE_WAKE, PHASE_SLEEP
from core.optimizers.NaturalWakeSleepOptimizer import get_diagonal_pad
from core.optimizers.NaturalWakeSleepOptimizerAlternate import NaturalWakeSleepOptimizerAlternate
from core.optimizers.ReweightedWakeSleepOptimizer import ReweightedWakeSleepOptimizer
from core.optimizers.WakeSleepOptimizer import WakeSleepOptimizer


class NaturalReweightedWakeSleepOptimizerAlternate(NaturalWakeSleepOptimizerAlternate):

    def __init__(self, **optimizer_kwargs):
        NaturalWakeSleepOptimizerAlternate.__init__(self, **optimizer_kwargs)

        self._sleep_balance = 0.5
        self._wake_q = 1.0 - self._sleep_balance
        self._sleep_q = self._sleep_balance

        self._qbaseline = tf.constant(0.)
        if optimizer_kwargs["q_baseline"]:
            self._qbaseline = tf.constant(1.) / tf.cast(self._n_samples, dtype=tf.float32)

    def compute_gradients(self, phase, *args, global_step=None, **kw):
        weights = []
        grads = []

        self._diagonal_pad = get_diagonal_pad(self._d_p, global_step=global_step)
        self._diagonal_cond = tf.less_equal(self._diagonal_pad, 10)

        self._nat_reg = [tf.constant(0.0)]

        if phase == PHASE_WAKE:
            hrw = self._model._hrw
            hgw = self._model._hgw
            # NATURAL GRADIENT PART
            hgs = self._model._hgs
            # END OF NATURAL GRADIENT PART

            self._individual_learning_rate = self.check_ilr(self._individual_learning_rate, hrw)

            weight_b = WakeSleepOptimizer._get_bias()  # Size of the last hidden layer
            weights = [weight_b]  # Size of the last hidden layer

            difference_b = hrw[-1][1] - hgw[-1][0].mean()

            # CALC WEIGHT PART
            imp_weights_p = ReweightedWakeSleepOptimizer.compute_normalized_weights(self, hr=hrw, hg=hgw)
            imp_weights_q = imp_weights_p - self._qbaseline
            # END OF CALC WEIGHT PART

            difference_b = tf.einsum("b,bv->bv", imp_weights_p, difference_b)
            difference_b = tf.reduce_sum(
                tf.reshape(difference_b, [self._n_samples, self._b_size, difference_b.shape.as_list()[1]]),
                axis=0)

            # NATURAL GRADIENT PART
            grad_b, _ = self._apply_fisher_multipliers(
                next_layer_distr_probs=hgs[-1][0].get_probs(),
                previous_layer_sample=None,
                difference_b=tf.reduce_mean(difference_b, axis=0),
                difference_w=None,
                global_step=global_step,
                layer=PHASE_WAKE + "B",
                weight_w=None,
                weight_b=weight_b)
            # END OF NATURAL GRADIENT PART

            grads = [
                tf.multiply(self._individual_learning_rate[-1], grad_b,
                            name="Wb{}".format(len(hrw) - 1))]

            for i in range(len(hrw) - 1)[::-1]:
                weight_w, weight_b = WakeSleepOptimizer._get_layer_vars("gen_net", "l_g_{}".format(i))
                weights += [weight_w, weight_b]

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

                # NATURAL GRADIENT PART
                grad_b, grad_w = self._apply_fisher_multipliers(
                    next_layer_distr_probs=hgs[i][0].get_probs(),
                    previous_layer_sample=hgs[i + 1][1],
                    difference_b=tf.reduce_mean(difference_b, axis=0),
                    difference_w=tf.reduce_mean(difference_w, axis=0),
                    global_step=global_step,
                    layer=PHASE_WAKE + str(i),
                    weight_w=weight_w,
                    weight_b=weight_b)
                # END OF NATURAL GRADIENT PART

                grads += [tf.multiply(self._individual_learning_rate[i], grad_w,
                                      name="Ww{}".format(i)),
                          tf.multiply(self._individual_learning_rate[i], grad_b,
                                      name="Wb{}".format(i))]

            # WAKE PHASE SLEEP
            for i in range(len(hrw) - 1):
                weight_w, weight_b = WakeSleepOptimizer._get_layer_vars("rec_net", "l_r_{}".format(i))
                weights += [weight_w, weight_b]

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

                # NATURAL GRADIENT PART
                grad_b, grad_w = self._apply_fisher_multipliers(
                    next_layer_distr_probs=hrw[i + 1][0].get_probs(),
                    previous_layer_sample=hrw[i][1],
                    difference_b=tf.reduce_mean(difference_b, axis=0),
                    difference_w=tf.reduce_mean(difference_w, axis=0),
                    global_step=global_step,
                    layer=PHASE_SLEEP + str(i),
                    weight_w=weight_w,
                    weight_b=weight_b)
                # END OF NATURAL GRADIENT PART

                grads += [
                    tf.multiply(self._wake_q * self._individual_learning_rate[i], grad_w,
                                name="WSw{}".format(i)),
                    tf.multiply(self._wake_q * self._individual_learning_rate[i], grad_b,
                                name="WSb{}".format(i))]

        elif phase == PHASE_SLEEP:
            # CLASSIC SLEEP
            hrs = self._model._hrs
            hgs = self._model._hgs
            # NATURAL GRADIENT PART
            hrw = self._model._hrw
            # END OF NATURAL GRADIENT PART

            self._individual_learning_rate = self.check_ilr(self._individual_learning_rate, hrs)

            for i in range(len(hrs) - 1):
                weight_w, weight_b = WakeSleepOptimizer._get_layer_vars("rec_net", "l_r_{}".format(i))
                weights += [weight_w, weight_b]

                difference_b = hgs[i + 1][1] - hrs[i + 1][0].mean()
                difference_w = tf.einsum("bu,bv->bvu", difference_b, hgs[i][1])

                # NATURAL GRADIENT PART
                grad_b, grad_w = self._apply_fisher_multipliers(
                    next_layer_distr_probs=hrw[i + 1][0].get_probs(),
                    previous_layer_sample=hrw[i][1],
                    difference_b=tf.reduce_mean(difference_b, axis=0),
                    difference_w=tf.reduce_mean(difference_w, axis=0),
                    global_step=global_step,
                    layer=PHASE_SLEEP + str(i) + "_",
                    weight_w=weight_w,
                    weight_b=weight_b)
                # END OF NATURAL GRADIENT PART

                grads += [
                    tf.multiply(self._sleep_q * self._individual_learning_rate[i], grad_w,
                                name="Sw{}".format(i)),
                    tf.multiply(self._sleep_q * self._individual_learning_rate[i], grad_b,
                                name="Sb{}".format(i))]

        else:
            raise ValueError("invalid value for phase '{}'".format(phase))

        lr = 1.
        if phase == PHASE_SLEEP:
            lr *= self._rescale_learning_rate

        regs = self.get_regularizers(weights)
        nat_regs = self._get_natural_regularizers(weights)

        grads_and_vars_not_none = [(tf.multiply(-lr, g, name="g_" + g.name.split(":")[0]) + r + nr, v) for (g, r, nr, v) in
                                   zip(grads, regs, nat_regs, weights) if g is not None]


        assert np.all([g.shape == v.shape for (g, v) in
                       grads_and_vars_not_none]), "The shapes of weights and gradients are not the same"

        return grads_and_vars_not_none

    def get_unnormalized_weigth_log(self, log_probs_p, log_probs_q):
        return log_probs_p - log_probs_q
