import tensorflow as tf
from tensorflow_probability import distributions as tfd

import sonnet as snt

import numpy as np

import pdb

from argo.core.network.Network import AbstractModule
from argo.core.utils.argo_utils import NUMTOL

class UnfusedCrossEntropyWithLogits(AbstractModule):

    def __init__(self, name="UnfusedCrossEntropyWithLogits"):
        super().__init__(name=name)

    @staticmethod
    def create_id(cost_fuction_kwargs):
        _id = "UCE"

        return _id

    def _build(self, prediction_model, drop_one_logit=False):

        y = prediction_model.y
        logits = prediction_model.logits

        n = logits.get_shape().as_list()[1]
        probabilities = tf.nn.softmax(logits)

        clipped_probabilities = tf.clip_by_value(probabilities, NUMTOL, 1 - NUMTOL)
        loss = tf.reduce_sum(-tf.one_hot(y, depth=n) * tf.log(clipped_probabilities), axis=1)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                                   tf.cast(y, dtype=tf.int64)),
                                          dtype=tf.float32))

        return loss, [[1 - accuracy]], [["error"]], [{"fileName": "error", "logscale-y": 1}]
