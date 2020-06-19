import tensorflow as tf
import sonnet as snt

def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens, activation,
                   training, prob_drop, momentum, renorm, renorm_momentum, renorm_clipping):

    for i in range(num_residual_layers):
        h_i = activation(h)

        h_i = snt.Conv2D(
            output_channels=num_residual_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="res3x3_%d" % i)(h_i)

        h_i = tf.layers.dropout(h_i, prob_drop, training=training)
        h_i = tf.layers.batch_normalization(h_i, training=training,
                                          momentum=momentum,
                                          renorm=renorm,
                                          renorm_momentum=renorm_momentum,
                                          renorm_clipping=renorm_clipping,
                                          name="bn_1_%d" % i)

        h_i = activation(h_i)

        h_i = snt.Conv2D(
            output_channels=num_hiddens,
            kernel_shape=(1, 1),
            stride=(1, 1),
            name="res1x1_%d" % i)(h_i)

        h_i = tf.layers.dropout(h_i, prob_drop, training=training)
        h_i = tf.layers.batch_normalization(h_i, training=training,
                                          momentum=momentum,
                                          renorm=renorm,
                                          renorm_momentum=renorm_momentum,
                                          renorm_clipping=renorm_clipping,
                                          name="bn_2_%d" % i)

        h += h_i

    return activation(h)

