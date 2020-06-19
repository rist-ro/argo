import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import NadamOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

from core.HM import PHASE_WAKE, PHASE_SLEEP
from argo.core.optimizers.NesterovConst import NesterovConst


class WakeSleepOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, **optimizer_kwargs):
        self._model = optimizer_kwargs["model"]

        self._individual_learning_rate = optimizer_kwargs["individual_learning_rate"]

        self._learning_rate = optimizer_kwargs["learning_rate"]
        self._rescale_learning_rate = optimizer_kwargs["rescale_learning_rate"]
        self._d_p = None
        self._n_reg = None

        post_optimizer = optimizer_kwargs["post_optimizer"] if "post_optimizer" in optimizer_kwargs else None
        if post_optimizer is None:
            self._post_optimizer = super()

        elif post_optimizer == "Momentum":
            self._post_optimizer = MomentumOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                     momentum=0.95,
                                                     use_locking=False,
                                                     name="MomentumOptimizer")

        elif post_optimizer == "RMSProp":
            self._post_optimizer = RMSPropOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                    decay=0.9,
                                                    epsilon=1e-5,
                                                    use_locking=False,
                                                    name="RMSPropOptimizer")

        elif post_optimizer == "Adam":
            self._post_optimizer = AdamOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                 beta1=0.9,
                                                 beta2=0.999,
                                                 epsilon=1e-8,
                                                 use_locking=False,
                                                 name="AdamOptimizer")
        elif post_optimizer == "Nadam":
            self._post_optimizer = NadamOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                 beta1=0.9,
                                                 beta2=0.999,
                                                 epsilon=1e-8,
                                                 use_locking=False,
                                                 name="NadamOptimizer")

        elif post_optimizer == "Nesterov":
            self._post_optimizer = MomentumOptimizer(learning_rate=optimizer_kwargs["learning_rate"],
                                                     momentum=0.95,
                                                     use_locking=False,
                                                     use_nesterov=True,
                                                     name="NesterovMomentumOptimizer")
        elif post_optimizer == "NesterovConst":
            self._post_optimizer = NesterovConst(model=self._model,
                                                 learning_rate=optimizer_kwargs["learning_rate"],
                                                 use_locking=False,
                                                 name="NesterovConstOptimizer")

        else:
            raise Exception("There is no such post optimizer defined. Must be: None, Adam, Momentum, RMSProp")

        super().__init__(self._learning_rate)

    def check_ilr(self, individual_learning_rate, hr):
        if isinstance(individual_learning_rate, list):
            assert len(individual_learning_rate) == len(
                hr), "Individual learning rates have to equal in length the number of layers, {} and {}".format(
                individual_learning_rate, len(hr))
            return list(map(float, individual_learning_rate))
        elif isinstance(individual_learning_rate, (int, float)):
            return [float(individual_learning_rate)] * len(hr)
        else:
            raise Exception("You gave an unexpected data type as Individual learning rates")

    def apply_gradients(self, grads_and_vars, global_step=None, name="WSOM"):
        return self._post_optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step, name=name)

    def compute_gradients(self, phase, *args, **kw):

        weights = []
        grads = []
        if phase == PHASE_WAKE:
            hrw = self._model._hrw
            hgw = self._model._hgw
            self._individual_learning_rate = self.check_ilr(self._individual_learning_rate, hrw)

            weights = [self._get_bias()]  # Size of the last hidden layer
            difference_b = hrw[-1][1] - hgw[-1][0].mean()
            grads = [tf.multiply(self._individual_learning_rate[-1], tf.reduce_mean(difference_b,
                                                                                    axis=0),
                                 name="Wb{}".format(len(hrw) - 1))]

            for i in range(len(hrw) - 1)[::-1]:
                weights += [*self._get_layer_vars("gen_net", "l_g_{}".format(i))]

                difference_b = hrw[i][1] - hgw[i][0].mean()
                difference_w = tf.einsum("bu,bv->bvu", difference_b, hrw[i + 1][1])

                grads += [tf.multiply(self._individual_learning_rate[i], tf.reduce_mean(difference_w, axis=0),
                                      name="Ww{}".format(i)),
                          tf.multiply(self._individual_learning_rate[i], tf.reduce_mean(difference_b, axis=0),
                                      name="Wb{}".format(i))]

        elif phase == PHASE_SLEEP:
            hrs = self._model._hrs
            hgs = self._model._hgs

            self._individual_learning_rate = self.check_ilr(self._individual_learning_rate, hrs)

            for i in range(len(hrs) - 1):
                weights += [*self._get_layer_vars("rec_net", "l_r_{}".format(i))]

                difference_b = hgs[i + 1][1] - hrs[i + 1][0].mean()
                difference_w = tf.einsum("bu,bv->bvu", difference_b, hgs[i][1])

                grads += [tf.multiply(self._individual_learning_rate[i], tf.reduce_mean(difference_w, axis=0),
                                      name="Sw{}".format(i)),
                          tf.multiply(self._individual_learning_rate[i], tf.reduce_mean(difference_b, axis=0),
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

    def get_regularizers(self, weights):
        regs = [0.0] * len(weights)
        if self._model.regularizers:
            loss = 0.0 + tf.add_n(self._model.regularizers, name="regularization")
            regs = tf.gradients(loss, weights)
        return regs

    @staticmethod
    def _get_layer_vars(phase, layer_name):
        layer_scope_name = ".*" + phase + "/" + layer_name + "/.*"
        ws = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=layer_scope_name + "/w:0")
        if not ws:
            raise Exception("found no weights in scope: %s" % layer_scope_name)
        elif len(ws) > 1:
            raise Exception("found more than one weight in scope: %s"
                            "\n%s" % (layer_scope_name, str(ws)))
        w = ws[0]

        bs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=layer_scope_name + "/b:0")
        if not bs:
            raise Exception("found no biases in scope: %s" % layer_scope_name)
        elif len(bs) > 1:
            raise Exception("found more than one bias in scope: %s"
                            "\n%s" % (layer_scope_name, str(bs)))
        b = bs[0]

        return w, b

    @staticmethod
    def _get_bias():
        bs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=".*/top_bias/linear/b:0")
        if not bs:
            raise Exception("found no prior bias")
        return bs[0]
