import tensorflow as tf
import tensorflow_probability as tfp
from .keras_utils import get_renorm_clipping, get_keras_activation
from .ArgoKerasModel import ArgoKerasModel

class BayesianResNet(ArgoKerasModel):
    """Constructs a ResNet model.
    """
    def __init__(self,
                 filters=[16, 16, 32, 32, 32],
                 kernels=[3, 3, 3, 3, 3],
                 strides=[2, 2, 2, 2, 2],
                 logits_size=None,
                 flipout=True,
                 renorm=False,
                 activation = ('relu', {}),
                 layer_kwargs={},
                 layer_kwargs_bayes={}):

        super().__init__(name='bayesian_vgg')
        self._activation_tuple = activation

        if flipout:
            Conv2D = tfp.layers.Convolution2DFlipout
            Dense = tfp.layers.DenseFlipout
        else:
            Conv2D = tfp.layers.Convolution2DReparameterization
            Dense = tfp.layers.DenseReparameterization

        renorm_clipping = None
        if renorm:
            renorm_clipping = get_renorm_clipping()

        self.blocks_list = []

        for i in range(len(kernels)):
            block = ResnetBlock(Conv2D,
                            filters[i],
                            kernels[i],
                            strides[i],
                            renorm,
                            renorm_clipping,
                            activation_tuple = activation,
                            **layer_kwargs_bayes)
            self.blocks_list.append(block)

        self._pooling = tf.keras.layers.AveragePooling2D(2,1)
        self._flat = tf.keras.layers.Flatten()
        # if logits_size is specified I add an extra Dense layer
        self._last_dense = None
        if logits_size is not None:
            self._last_dense = Dense(
                                logits_size,
                                **layer_kwargs_bayes)


    def call(self, inputs, training=False, **extra_kwargs):
        net = inputs
        for block in self.blocks_list:
            net = block(net, training=training)

        net = self._pooling(net)
        net = self._flat(net)

        if self._last_dense:
            net = self._last_dense(net)

        return net



class ResnetBlock(tf.keras.Model):
    def __init__(self, Conv2D, filters, kernel, stride,
                 renorm, renorm_clipping,
                 activation_tuple= ('relu', {}),
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='resnet_block')

        self.conv2a = Conv2D(
                        filters,
                        kernel,
                        padding='same',
                        **layer_kwargs_bayes)

        self.bn2a = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        self._act2a = get_keras_activation(activation_tuple)

        self.conv2b = Conv2D(
                        filters,
                        kernel,
                        padding='same',
                        strides=stride,
                        **layer_kwargs_bayes)

        self.bn2b = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        self._act2b = get_keras_activation(activation_tuple)
        # self.conv2c = None
        # if stride>1:
        self.conv2c = Conv2D(filters,
                             1,
                             padding='same',
                             strides=stride,
                             **layer_kwargs_bayes)

        self._act2c = None
        if final_activation:
            self._act2c = get_keras_activation(activation_tuple)
            self.bn2c = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)

    def call(self, input_tensor, training=False, **extra_kwargs):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = self._act2a(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self._act2b(x)

        # if self.conv2c:
        x_sh = self.conv2c(input_tensor)
        # else:
        #     x_sh = input_tensor

        output = x + x_sh
        if self._act2c:
            output = self.bn2c(output, training=training)
            output = self._act2c(output)
        return output


