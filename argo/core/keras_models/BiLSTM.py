import tensorflow as tf

# layer_kwargs = {
#            "kernel_initializer" : tf.glorot_normal_initializer(),
#            "bias_initializer": tf.initializers.constant(value=1.),
#
#            "kernel_regularizer": tf.contrib.layers.l2_regularizer(scale=1e-5),
#            "bias_regularizer": tf.contrib.layers.l2_regularizer(scale=1e-5),
#
#            "activity_regularizer": None
#         }

def BiLSTM(output_size, layer_kwargs = {}, layer_kwargs_bayes = {}):

    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, **layer_kwargs)),
        tf.keras.layers.Dense(64, activation='relu', **layer_kwargs),
        tf.keras.layers.Dense(output_size, activation=None, **layer_kwargs)
    ])

    return model