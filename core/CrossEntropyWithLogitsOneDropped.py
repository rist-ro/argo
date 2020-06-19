import tensorflow as tf
from tensorflow_probability import distributions as tfd

import sonnet as snt

import numpy as np

import pdb

from argo.core.network.Network import AbstractModule
from argo.core.utils.argo_utils import NUMTOL, tf_f1_score


class CrossEntropyWithLogitsOneDropped(AbstractModule):

    def __init__(self, multiclass_metrics=False, name="CrossEntropyWithLogitsOneDropped"):  # drop_one_logit=0,
        super().__init__(name=name)
        self._multiclass_metrics = multiclass_metrics
        # self._drop_one_logit = drop_one_logit

    @staticmethod
    def create_id(cost_fuction_kwargs):

        _id = "CEdrop"  # + str(cost_fuction_kwargs.get("drop_one_logit",0))

        return _id

    def _build(self, prediction_model, drop_one_logit=False):

        # dims = np.prod(ae.x_shape)
        # this has changed since some models can feed data with different shapes for train and eval
        # if this break ask me.. (Riccardo) -> I fixed, self.raw_x is replaces by ae.raw_x

        '''
        flatten = snt.BatchFlatten()(prediction_model.logits)
        dim_output = prediction_model.dataset.n_labels


        # use the same default values as for the network
        default_weights_init = prediction_model._network._default_weights_init
        default_bias_init = prediction_model._network._default_bias_init
        initializers = {}
        initializers['w'] = default_weights_init
        initializers['b'] = default_bias_init

        default_weights_reg = prediction_model._network._default_weights_reg
        default_bias_reg = prediction_model._network._default_bias_reg
        regularizers = {}
        regularizers['w'] = default_weights_reg
        regularizers['b'] = default_bias_reg

        logits = snt.Linear(dim_output, initializers=initializers, regularizers=regularizers)(flatten)
        '''

        y = prediction_model.y
        logits = prediction_model.logits
        n_labels = logits.shape[1]

        if drop_one_logit:
            n = logits.get_shape().as_list()[1]

            logits_add_one_logit = tf.concat([logits,
                                              tf.zeros_like(tf.slice(logits, [0, 0], [-1, 1]))],
                                             1)

            probabilities = tf.nn.softmax(logits_add_one_logit)
            probabilities_sliced = tf.slice(probabilities, [0, 0], [-1, n])
            new_probabilities = tf.concat([probabilities_sliced,
                                           tf.reshape(1 - tf.reduce_sum(probabilities_sliced, axis=1), [-1, 1])],
                                          1)

            clipped_probabilies = tf.clip_by_value(new_probabilities, NUMTOL, 1 - NUMTOL)
            # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
            loss_per_sample = tf.reduce_sum(-tf.one_hot(y, depth=n + 1) * tf.log(clipped_probabilies), axis=1)
            loss = tf.reduce_mean(loss_per_sample)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_add_one_logit, axis=1),
                                                       tf.cast(y, dtype=tf.int64)),
                                              dtype=tf.float32))

        else:

            loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y, tf.int32),
                                                                             logits=logits)
            loss = tf.reduce_mean(loss_per_sample)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                                       tf.cast(y, dtype=tf.int64)),
                                              dtype=tf.float32))

        nodes_to_log = [[1 - accuracy],
                        [accuracy]]
        nodes_to_log_names = [["error"], ["accuracy"]]
        nodes_to_log_filenames = [{"fileName": "error", "logscale-y": 1},
                                  {"fileName": "accuracy"}]

        if self._multiclass_metrics:
            y_pred = tf.one_hot(tf.argmax(logits, axis=1), n_labels)
            y_true = tf.one_hot(y, n_labels)
            f1_micro, f1_macro, f1_weighted = tf_f1_score(y_true, y_pred)

            auc, auc_update = tf.metrics.auc(
                labels=tf.cast(y_true, dtype=tf.float32),
                predictions=tf.nn.softmax(logits)
            )

            nodes_to_log += [[auc_update],
                             [f1_micro, f1_macro, f1_weighted]]

            nodes_to_log_names += [["auc"], ["f1_micro", "f1_macro", "f1_weighted"]]
            nodes_to_log_filenames += [
                {"fileName": "auc"},
                {"fileName": "f1_score"}
                # {"fileName": "f1_micro"},
                # {"fileName": "f1_macro"},
                # {"fileName": "f1_weighted"}
            ]

        return loss, loss_per_sample, nodes_to_log, nodes_to_log_names, nodes_to_log_filenames


