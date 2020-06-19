import tensorflow as tf
import tensorflow_probability as tfp
from prediction.argo.core.keras_models.keras_utils import get_kwargs_bayes


def bayesian_resnet(input_shape,
                    logits_size=2):
    """Constructs a ResNet18 model.
    Args:
    input_shape: A `tuple` indicating the Tensor shape.
    output_size: `int` representing the output size.
    kernel_posterior_scale_mean: Python `int` number for the kernel
      posterior's scale (log variance) mean. The smaller the mean the closer
      is the initialization to a deterministic network.
    kernel_posterior_scale_stddev: Python `float` number for the initial kernel
      posterior's scale stddev.
      ```
      q(W|x) ~ N(mu, var),
      log_var ~ N(kernel_posterior_scale_mean, kernel_posterior_scale_stddev)
      ````
    kernel_posterior_scale_constraint: Python `float` number for the log value
      to constrain the log variance throughout training.
      i.e. log_var <= log(kernel_posterior_scale_constraint).
    Returns:
    tf.keras.Model.
    """

    filters = [64, 128, 256, 512]
    kernels = [3, 3, 3, 3]
    strides = [1, 2, 2, 2]
    kwargs_bayes = get_kwargs_bayes()

    image = tf.keras.layers.Input(shape=input_shape, dtype='float32')
    x = tfp.layers.Convolution2DFlipout(
          64,
          3,
          strides=1,
          padding='same',
          **kwargs_bayes)(image)

    for i in range(len(kernels)):
        x = _resnet_block(
                    x,
                    filters[i],
                    kernels[i],
                    strides[i],
                    **kwargs_bayes)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.AveragePooling2D(4, 1)(x)
    x = tf.keras.layers.Flatten()(x)

    x = tfp.layers.DenseFlipout(
              logits_size,
              **kwargs_bayes)(x)

    model = tf.keras.Model(inputs=image, outputs=x, name='resnet18')
    return model


def _resnet_block(x, filters, kernel, stride, **kwargs_bayes):
    """Network block for ResNet."""
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    if stride != 1 or filters != x.shape[1]:
        shortcut = _projection_shortcut(x, filters, stride, **kwargs_bayes)
    else:
        shortcut = x

    x = tfp.layers.Convolution2DFlipout(
          filters,
          kernel,
          strides=stride,
          padding='same',
          **kwargs_bayes)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tfp.layers.Convolution2DFlipout(
          filters,
          kernel,
          strides=1,
          padding='same',
          **kwargs_bayes)(x)
    x = tf.keras.layers.add([x, shortcut])
    return x


def _projection_shortcut(x, out_filters, stride, **kwargs_bayes):
    x = tfp.layers.Convolution2DFlipout(
              out_filters,
              1,
              strides=stride,
              padding='valid',
              **kwargs_bayes)(x)
    return x



class ResnetBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, conv_type="base"):
        super().__init__(name='resnet_block')

        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False, **extra_kwargs):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)
