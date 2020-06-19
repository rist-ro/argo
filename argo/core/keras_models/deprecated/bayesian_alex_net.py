import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial

def bayesian_alex_net(
            filters=[32, 32, 32, 32, 32],  # [32, 64, 64, 128, 128],
            kernels=[7,5,3,3,3],
            strides=[4, 2, 2, 2, 2],
            renorm = False,
            logits_size=None,
            flipout = True,
            pooling="avg",
            layer_kwargs = {},
            layer_kwargs_bayes = {}):

    if flipout:
        Conv2D = tfp.layers.Convolution2DFlipout
        Dense = tfp.layers.DenseFlipout
    else:
        Conv2D = tfp.layers.Convolution2DReparameterization
        Dense = tfp.layers.DenseReparameterization

    pooling_choices = ["max", "avg"]
    if pooling not in pooling_choices:
        raise ValueError("pooling must be in {:}, instead {:} found.".format(pooling_choices, pooling))

    if pooling=="max":
        Pool = tf.keras.layers.MaxPooling2D
    elif pooling=="avg":
        Pool = tf.keras.layers.AveragePooling2D

    activation = partial(tf.nn.leaky_relu)

    _cnn = tf.keras.Sequential()

    for i in range(len(filters)):
        _cnn.add(Conv2D(filters[i],
                     kernel_size=kernels[i],
                     padding="SAME",
                     activation=activation,
                     **layer_kwargs_bayes))
        if i == 0 or i == 1 or i == 4:
            _cnn.add(Pool(pool_size=[2, 2], strides=strides[i], padding="SAME"))


    _cnn.add(Conv2D(32,
                    kernel_size=1,
                     padding="SAME",
                     activation=activation,
                     **layer_kwargs_bayes))

    # _cnn.add(Conv2D(32,
    #                  kernel_size=1,
    #                  padding="SAME",
    #                  activation=activation,
    #                  **layer_kwargs_bayes))

    if logits_size is not None:
        _cnn.add(tf.keras.layers.Flatten())
        _cnn.add(tf.keras.layers.Dense(logits_size,
            **layer_kwargs_bayes))

    return _cnn
