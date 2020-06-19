import numpy as np
import tensorflow as tf

from core.optimizers.NaturalWakeSleepOptimizer import NaturalWakeSleepOptimizer
from core.optimizers.linalg.straight_inverse import _true_fisher_inverse, _damped_fisher_inverse


class NaturalWakeSleepOptimizerAlternate(NaturalWakeSleepOptimizer):

    def __init__(self, **optimizer_kwargs):
        NaturalWakeSleepOptimizer.__init__(self, **optimizer_kwargs)

        self._k_step_update = optimizer_kwargs["k_step_update"]
        self._saves = {}

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

        if previous_layer_sample is None:  # just the bias
            weight_b = tf.cast(weight_b, dtype=tf.float64)
            fisher_weights_reduced = self._alternate_node(
                value_fn=lambda: tf.reduce_sum(get_fisher_weights(), axis=0),
                shape=[next_layer_distr_probs.shape.as_list()[-1]],
                name=layer + "B",
                global_step=global_step,
                dtype=next_layer_distr_probs.dtype)

            alpha = self._diagonal_pad
            if self._d_p is None or self._d_p == 0.0:
                grads_b = difference_b / fisher_weights_reduced
            else:
                grads_b = self._damping_multiplier(
                    lambda: difference_b * (1.0 + alpha) / (alpha + fisher_weights_reduced),
                    lambda: difference_b)
            grads_b = tf.cast(grads_b, dtype=orig_dtype)
            self._nat_reg.append(tf.cast(self._n_reg * weight_b * alpha * weight_b, dtype=orig_dtype)[0])

            return grads_b, None

        difference_w = tf.cast(difference_w, dtype=tf.float64)
        bias_pad = tf.constant([[0, 0], [0, 1]])

        previous_layer_sample_concat = tf.pad(previous_layer_sample, bias_pad, "CONSTANT", constant_values=1)

        previous_layer_sample_transposed = tf.transpose(previous_layer_sample_concat, perm=[1, 0])

        diff_concat = tf.concat([difference_w, tf.reshape(difference_b, [1, -1])], axis=0)
        # if self._n_reg:
        weight_concat = tf.concat([weight_w, tf.reshape(weight_b, [1, -1])], axis=0)
        # else:
        #     weight_concat = None

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
                                      layer):  # , choice=False):
        weights = tf.cast(weights, dtype=tf.float64)
        weight_concat = tf.cast(weight_concat, dtype=tf.float64)
        previous_layer = tf.cast(previous_layer, dtype=tf.float64)
        alpha = self._diagonal_pad
        alpha = tf.cast(alpha, dtype=tf.float64)

        previous_layer_transpose = tf.transpose(previous_layer, perm=[1, 0])
        if self._d_p is None or self._d_p == 0.0:
            inv_fisher = self._alternate_node(
                value_fn=lambda: _true_fisher_inverse(U=previous_layer, Q=weights, U_T=previous_layer_transpose),
                shape=[difference.shape.as_list()[0], difference.shape.as_list()[-1], difference.shape.as_list()[-1]],
                name=layer + "MIT",
                global_step=global_step,
                dtype=previous_layer.dtype)

            inverse_x_dif = tf.einsum('lkn,ln->lk', inv_fisher, difference)
        else:
            def _calc_sh():
                if (self._model.batch_size['train'] * self._model.samples < difference.shape[-1]):  # ^ choice:

                    C = weights
                    C_inv = tf.linalg.diag(1 / C)

                    grads = difference

                    k_len = self._model.batch_size['train'] * self._model.samples

                    u = self._alternate_node(
                        value_fn=lambda: previous_layer,
                        shape=[previous_layer.shape.as_list()[0], k_len],
                        name=layer + "U",
                        global_step=global_step,
                        dtype=previous_layer.dtype)

                    v_T = tf.transpose(u, perm=[1, 0])

                    m_inner_inv = self._alternate_node(
                        value_fn=lambda: tf.linalg.inv(alpha * C_inv + tf.einsum('ij,jk->ik', v_T, u)),
                        shape=[grads.shape.as_list()[0], k_len, k_len],
                        name=layer + "MII",
                        global_step=global_step,
                        dtype=previous_layer.dtype)

                    M2 = tf.einsum('ij,lj->li', u,
                                   tf.einsum('lij,lj->li', m_inner_inv, tf.einsum('ik,lk->li', v_T, grads)))

                    M = grads - M2

                    inverse_x_dif = ((1.0 + alpha) / alpha) * M

                    inverse_x_dif.set_shape(grads.shape.as_list())

                else:
                    inv_fisher = self._alternate_node(
                        value_fn=lambda: _damped_fisher_inverse(U=previous_layer, Q=weights,
                                                                U_T=previous_layer_transpose,
                                                                alpha=self._diagonal_pad),
                        shape=[difference.shape.as_list()[0], difference.shape.as_list()[-1],
                               difference.shape.as_list()[-1]],
                        name=layer + "MI",
                        global_step=global_step,
                        dtype=previous_layer.dtype)
                    inverse_x_dif = tf.einsum('lkn,ln->lk', inv_fisher, difference)
                return inverse_x_dif

            inverse_x_dif = self._damping_multiplier(_calc_sh, lambda: difference)

        self._get_natural_regs(U=previous_layer, Q=weights, W=weight_concat, orig_dtype=orig_dtype)
        inverse_x_dif = tf.cast(inverse_x_dif, dtype=orig_dtype)
        return inverse_x_dif

    def _alternate_node(self, value_fn, shape, name, dtype, global_step):

        self._register_node(shape, name, dtype)

        def tru():
            node = tf.identity(value_fn())
            return self._update_node(node, name)

        def fal():
            node = tf.identity(self._get_node_by_varname(name))
            return node

        node = tf.cond(self._global_step_cond(global_step=global_step), tru, fal)
        return node

    def _global_step_cond(self, global_step):
        if self._k_step_update <= 0:
            return tf.constant(True)
        return tf.logical_or(tf.equal(global_step, 0), tf.equal(global_step % self._k_step_update, 0))

    def _update_node(self, var, var_name, validate_shape=True):
        if self._check_node(var_name):
            new_value = tf.assign(self._saves[var_name], var, validate_shape=validate_shape)
            return new_value
        else:
            raise ValueError("Var name is accessed before creation: '{}'".format(var_name))

    def _register_node(self, exp_shape, var_name, dtype, validate_shape=True):
        if self._check_node(var_name):
            raise ValueError("Var name is already created: '{}'".format(var_name))
        else:
            if validate_shape == True:
                self._saves[var_name] = tf.Variable(np.zeros(exp_shape), expected_shape=exp_shape, dtype=dtype)
            else:
                self._saves[var_name] = tf.Variable([], shape=None, validate_shape=False, dtype=dtype)

    def _check_node(self, var_name):
        return var_name in self._saves.keys()

    def _get_node_by_varname(self, var_name):
        if self._check_node(var_name):
            return self._saves[var_name]
        else:
            raise ValueError("Var name is accessed before assignment: '{}'".format(var_name))
