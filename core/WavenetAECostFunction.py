import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule


class WavenetAECostFunction(AbstractModule):

    def __init__(self, name="WNAE_L2"):
        super().__init__(name=name)
        self.name = name

    def create_id(self):
        _id = self.name

        return _id

    def _build(self, wavenet_ae):

        # x_indices = tf.cast(tf.reshape(wavenet_ae._network.x_quantized, [-1]), tf.int32) + 128
        x_indices = tf.cast(wavenet_ae._network.x_quantized, tf.int32) + 128
        x_indices = tf.squeeze(x_indices, axis=-1)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=wavenet_ae.x_reconstruction_logits_tf, labels=x_indices,
                                                           name='nll'), name='loss')

        return loss, [], []

    def loss_per_sample(self, reconstr_logits, x_target_quantized):
        x_target_quantized = tf.cast(x_target_quantized, tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reconstr_logits,
                                                           labels=x_target_quantized,
                                                           name='nll')

        return loss
