import numpy as np
import tensorflow as tf

from core.HM import PHASE_WAKE, PHASE_SLEEP
from core.optimizers.WakeSleepOptimizer import WakeSleepOptimizer
from core.optimizers.linalg.straight_inverse import _true_fisher_inverse, _damped_fisher_inverse
from core.optimizers.linalg.woodberry import _optimized_woodberry


def get_diagonal_pad(diagonal_pad, global_step):
    dp = None

    if diagonal_pad is None:
        pass

    elif isinstance(diagonal_pad, (int, float)):
        dp = tf.constant(float(diagonal_pad), dtype=tf.float64)

    elif isinstance(diagonal_pad, tuple):
        dp_min, dp_name, dp_kwargs = diagonal_pad
        dp_kwargs = dp_kwargs.copy()
        dp_method = getattr(tf.train, dp_name)
        dp_kwargs.update({
            "global_step": global_step})
        dp = dp_min + dp_method(**dp_kwargs)

    # instantiate lr node if lr is None and diagonal_pad is a dict at this point
    if dp is None and isinstance(diagonal_pad, dict):
        if not 0 in diagonal_pad:
            raise ValueError(
                "diagonal_pad schedule must specify, learning rate for step 0. Found schedule: %s" % diagonal_pad)

        dp = tf.constant(diagonal_pad[0])
        global_step = tf.train.get_or_create_global_step()
        for key, value in diagonal_pad.items():
            dp = tf.cond(
                tf.less(global_step, key), lambda: dp, lambda: tf.constant(value))
        tf.summary.scalar("diagonal_pad", dp)

    if dp is None:
        raise Exception("oops, something went wrong... could not process diagonal_pad {}".format(str(diagonal_pad)))

    return tf.identity(dp, name="diagonal_pad")


class NaturalWakeSleepOptimizer(WakeSleepOptimizer):

    def __init__(self, **optimizer_kwargs):
        WakeSleepOptimizer.__init__(self, **optimizer_kwargs)

        self._d_p = optimizer_kwargs["diagonal_pad"]
        self._n_reg = (optimizer_kwargs["natural_reg"] if "natural_reg" in optimizer_kwargs and float(
            optimizer_kwargs["natural_reg"]) > 0 else 0)
        self._b_size = self._model.b_size
        self._n_samples = self._model.n_z_samples
        self._n_z_length = self._n_samples * self._b_size

    def compute_gradients(self, phase, *args, global_step=None, **kw):

        weights = []
        grads = []

        self._diagonal_pad = get_diagonal_pad(self._d_p, global_step=global_step)
        self._diagonal_cond = tf.less_equal(self._diagonal_pad, 100.0)
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

            # NATURAL GRADIENT PART
            grad_b, _ = self._apply_fisher_multipliers(
                next_layer_distr_probs=hgs[-1][0].get_probs(),
                previous_layer_sample=None,
                difference_b=tf.reduce_mean(difference_b, axis=0),
                difference_w=None,
                global_step=global_step,
                layer=PHASE_WAKE + str(len(hrw) - 1) + "_",
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

                # NATURAL GRADIENT PART
                grad_b, grad_w = self._apply_fisher_multipliers(
                    next_layer_distr_probs=hgs[i][0].get_probs(),
                    previous_layer_sample=hgs[i + 1][1],
                    difference_b=tf.reduce_mean(difference_b, axis=0),
                    difference_w=tf.reduce_mean(difference_w, axis=0),
                    global_step=global_step,
                    layer=PHASE_WAKE + str(i) + "_",
                    weight_w=weight_w,
                    weight_b=weight_b)
                # END OF NATURAL GRADIENT PART

                grads += [tf.multiply(self._individual_learning_rate[i], grad_w,
                                      name="Ww{}".format(i)),
                          tf.multiply(self._individual_learning_rate[i], grad_b,
                                      name="Wb{}".format(i))]

        elif phase == PHASE_SLEEP:
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

                grads += [tf.multiply(self._individual_learning_rate[i], grad_w,
                                      name="Sw{}".format(i)),
                          tf.multiply(self._individual_learning_rate[i], grad_b,
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

    def _apply_fisher_multipliers(self, next_layer_distr_probs, previous_layer_sample, difference_b, difference_w,
                                  global_step, layer, weight_w=None, weight_b=None):
        # The formula is F=E[-q*(1-q)[h|1]^T[h|1]]

        assert not np.bitwise_xor(previous_layer_sample is not None,
                                  difference_w is not None), "In the case of the bias there's no grad_w and previous layer"

        orig_dtype = next_layer_distr_probs.dtype
        next_layer_distr_probs = tf.cast(next_layer_distr_probs, dtype=tf.float64)
        difference_b = tf.cast(difference_b, dtype=tf.float64)

        def get_fisher_weights():
            weights = (next_layer_distr_probs * (1 - next_layer_distr_probs)) / tf.cast(self._n_z_length,
                                                                                        dtype=tf.float64)

            if len(weights.shape) == 1:
                weights = tf.expand_dims(weights, axis=0)

            return weights

        if previous_layer_sample is None:
            alpha = self._diagonal_pad
            if self._d_p is None or self._d_p == 0.0:
                fisher_weights_reduced = tf.reduce_sum(get_fisher_weights(), axis=0)
                grads_b = difference_b / fisher_weights_reduced
            else:
                grads_b = self._damping_multiplier(
                    lambda: difference_b * (1.0 + alpha) / (alpha + tf.reduce_sum(get_fisher_weights(), axis=0)),
                    lambda: difference_b)
            grads_b = tf.cast(grads_b, dtype=orig_dtype)
            self._nat_reg.append(tf.cast(self._n_reg * weight_b * alpha * weight_b, dtype=orig_dtype)[0])
            return grads_b, None

        difference_w = tf.cast(difference_w, dtype=tf.float64)
        bias_pad = tf.constant([[0, 0], [0, 1]])

        previous_layer_sample_concat = tf.pad(previous_layer_sample, bias_pad, "CONSTANT", constant_values=1)

        previous_layer_sample_transposed = tf.transpose(previous_layer_sample_concat, perm=[1, 0])

        diff_concat = tf.concat([difference_w, tf.reshape(difference_b, [1, -1])], axis=0)

        weight_concat = tf.concat([weight_w, tf.reshape(weight_b, [1, -1])], axis=0)

        grads_w_b_concat = self._multiply_grads_by_fisher_inv(weight_concat=weight_concat,
                                                              difference=tf.transpose(diff_concat, perm=[1, 0]),
                                                              weights=tf.transpose(get_fisher_weights(), perm=[1, 0]),
                                                              previous_layer=previous_layer_sample_transposed,
                                                              global_step=global_step, orig_dtype=orig_dtype,
                                                              layer=layer)

        grads_w_b_concat = tf.transpose(grads_w_b_concat, perm=[1, 0])
        assert grads_w_b_concat.shape == diff_concat.shape, "Shapes of gradients pre and post multiplication are not equal"

        grads_w, grads_b = tf.split(grads_w_b_concat, [difference_w.shape.as_list()[0], 1], 0)

        grads_b = tf.reshape(grads_b, [-1])

        assert grads_b.shape == difference_b.shape, "Shapes of gradients pre and post multiplication are not equal for b"
        assert grads_w.shape == difference_w.shape, "Shapes of gradients pre and post multiplication are not equal for W"

        return grads_b, grads_w

    def _multiply_grads_by_fisher_inv(self, weight_concat, difference, weights, previous_layer, global_step, orig_dtype,
                                      layer):
        difference = tf.cast(difference, dtype=tf.float64)
        weight_concat = tf.cast(weight_concat, dtype=tf.float64)
        weights = tf.cast(weights, dtype=tf.float64)
        previous_layer = tf.cast(previous_layer, dtype=tf.float64)
        alpha = self._diagonal_pad
        alpha = tf.cast(alpha, dtype=tf.float64)

        previous_layer_transpose = tf.transpose(previous_layer, perm=[1, 0])

        if self._d_p is None or self._d_p == 0.0:
            inv_fisher = _true_fisher_inverse(U=previous_layer, Q=weights, U_T=previous_layer_transpose)
            inverse_x_dif = tf.einsum('lkn,ln->lk', inv_fisher, difference)
        else:
            def inv_fn():
                if (self._model.batch_size['train'] * self._model.samples < difference.shape[-1]):
                    inverse_x_dif = _optimized_woodberry(U=previous_layer, C=weights,
                                                         V_T=previous_layer_transpose,
                                                         alpha=alpha,
                                                         grads=difference, layer=layer)

                else:
                    inv_fisher = _damped_fisher_inverse(U=previous_layer, Q=weights, U_T=previous_layer_transpose,
                                                        alpha=alpha)

                    inverse_x_dif = tf.einsum('lkn,ln->lk', inv_fisher, difference)
                return inverse_x_dif

            inverse_x_dif = self._damping_multiplier(inv_fn, alternate_fn=lambda: difference)

        self._get_natural_regs(U=previous_layer, Q=weights, W=weight_concat, orig_dtype=orig_dtype)
        inverse_x_dif = tf.cast(inverse_x_dif, dtype=orig_dtype)
        return inverse_x_dif

    def _damping_multiplier(self, node_fn, alternate_fn):
        def tru():
            nody = tf.identity(node_fn())
            return nody

        def fal():
            nody = tf.identity(alternate_fn())
            return nody

        nody = tf.cond(self._diagonal_cond, tru, fal)
        return nody

    def _get_natural_regs(self, U, Q, W, orig_dtype):
        if self._n_reg > 0:
            U_T = tf.transpose(U, perm=[1, 0])
            W_T = tf.transpose(W, perm=[1, 0])
            regs = self._n_reg * tf.einsum('nk,kn->nk', W_T,
                                           tf.einsum('kl,ln->kn', U,
                                                     tf.einsum('njl,ln->jn', tf.linalg.diag(Q),
                                                               tf.einsum('lk,kn->ln',
                                                                         U_T, W))))
            loss = 0.0 + tf.reduce_sum(regs, name="nat_reg")
            self._nat_reg.append(tf.cast(loss, dtype=orig_dtype))


    def _get_natural_regularizers(self, weights):
        regs = [0.0] * len(weights)
        if self._n_reg > 0:
            loss = 0.0 + tf.add_n(self._nat_reg, name="natural_regularization")
            regs = tf.gradients(loss, weights)
        return regs