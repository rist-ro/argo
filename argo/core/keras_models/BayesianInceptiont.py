import tensorflow as tf
import tensorflow_probability as tfp
from .keras_utils import get_renorm_clipping
from keras.layers.merge import concatenate
#from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D
class BayesianInception(tf.keras.Model):
    """Constructs a ResNet model.
    """
    def __init__(self,
                 filters=[16, 32],
                 kernels=[5, 5],
                 strides=[1, 1],
                 logits_size=None,
                 flipout=True,
                 renorm=False,
                 pooling="avg",
                 activation = ('relu', {}),
                 layer_kwargs={},
                 layer_kwargs_bayes={}):

        super().__init__(name='bayesian_vgg')
        activation_name, activation_kwargs = activation

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

        renorm_clipping = None
        if renorm:
            renorm_clipping = get_renorm_clipping()


        def blockA(self):
            return ResnetBlockA_Inception(Conv2D,
                renorm,
                renorm_clipping,
                activation_name = activation_name,
                activation_kwargs = activation_kwargs,
                **layer_kwargs_bayes)
        def resnet_block(self):
            return ResnetBlock(Conv2D,
                renorm,
                renorm_clipping,
                activation_name = activation_name,
                activation_kwargs = activation_kwargs,
                **layer_kwargs_bayes)

        def blockB(self):
            return ResnetBlockB_Inception(Conv2D,
                renorm,
                renorm_clipping,
                activation_name = activation_name,
                activation_kwargs = activation_kwargs,
                **layer_kwargs_bayes)

        def blockC(self):
            return ResnetBlockC_Inception(Conv2D,
                renorm,
                renorm_clipping,
                activation_name = activation_name,
                activation_kwargs = activation_kwargs,
                **layer_kwargs_bayes)

        def pool_apply(self,pooling,stride_pool):
            return  Pool(pooling,stride_pool)

        #self._pooling = tf.keras.layers.AveragePooling2D(2,1)
        self._flat = tf.keras.layers.Flatten()
        # if logits_size is specified I add an extra Dense layer
        self._last_dense = None
        if logits_size is not None:
            self._last_dense = Dense(
                                logits_size,
                                **layer_kwargs_bayes)


    def call(self, inputs, training=False, **extra_kwargs):
        net = inputs
        for _ in range(4):
            net=self.blockA(net)
        net=self.resnet_block(net)
        for _ in range(6):
            net=self.blockB(net)
        net=self.pool_apply(3,2,poolname=AveragePooling2D)(net)
        net=self.resnet_block(net)
        for _ in range(3):
            net=self.blockC(net)
        net = self._flat(net)
        if self._last_dense:
            net = self._last_dense(net)
        return net

# def conv2d_bn(self,
#               filters,
#               num_row,
#               num_col,
#               padding='same',
#               strides=(1, 1),
#               name=None):
#             return Conv2D(filters,
#                         (num_row, num_col),
#                         strides=strides,
#                         padding=padding,
#                         **layer_kwargs_bayes)
#
# def batch_bn(self,renorm,renorm_clipping):
#     return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
# def activation(self,activation_name)
#     return tf.keras.layers.Activation(activation_name, **activation_kwargs)



class ResnetBlock(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='resnet_block')

        def conv2d_(self,
                      filters,
                      num_row,
                      num_col,
                      padding='same',
                      strides=(1, 1),
                      name=None):
            return Conv2D(filters,
                        (num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn(self):
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act(self):
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(self,pooling,stride_pool):
            return  Pool(pooling,stride_pool)

    def call(self, input_tensor, training=False, **extra_kwargs):

        branch_pool = self.pool_apply(3,2,poolname=AveragePooling2D)(input_tensor)

        branch5 = self.conv2d_(256, 1, 1)(input_tensor)
        branch5 = self._act2b(self.batch_bn(training=training)(branch5))
        branch5 = self.conv2d_(64, 3, 2)(branch5)
        branch5 = self._act2b(self.batch_bn(training=training)(branch5))

        branch3 = self.conv2d_(256, 1, 1)(x)
        branch3 = self._act2b(self.batch_bn(training=training)(branch3))
        branch3 = self.conv2d_(256, 3, 1)(branch3)
        branch3 = self._act2b(self.batch_bn(training=training)(branch3))
        branch3 = self.conv2d_(256, 3, 2)(branch3)
        branch3 = self._act2b(self.batch_bn(training=training)(branch3))

        output = concatenate([branch3, branch5,branch_pool], axis=3)
        return output

class ResnetBlockA_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='ResnetBlockA_Inception')

        def conv2d_(self,
                      filters,
                      num_row,
                      num_col,
                      padding='same',
                      strides=(1, 1),
                      name=None):
            return Conv2D(filters,
                        (num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn(self):
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act(self):
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(self,pooling,stride_pool):
            return  Pool(pooling,stride_pool)

        self.conv2b =

    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self.conv2d_(32, 1, 1)(input_tensor)
        branch1 = self._act2b(self.batch_bn(training=training)(branch1))

        branch2 = self.conv2d_(32, 1, 1)(input_tensor)
        branch2 = self._act2b(self.batch_bn(training=training)(branch2))
        branch2 = self.conv2d_(32, 3, 1)(branch2)
        branch2 = self._act2b(self.batch_bn(training=training)(branch2))

        branch3 = self.conv2d_(32, 1, 1)(input_tensor)
        branch3 = self._act2b(self.batch_bn(training=training)(branch3))
        branch3 = self.conv2d_(48, 3, 1)(branch3)
        branch3 = self._act2b(self.batch_bn(training=training)(branch3))
        branch3 = self.conv2d_(64, 3, 1)(branch3)
        branch3 = self._act2b(self.batch_bn(training=training)(branch3))

        mix = concatenate([branch1, branch2, branch3], axis=3)

        final = self.conv2d_(384, 1, 1)(mix)
        final = self._act2b(self.batch_bn(training=training)(final))
        return final


class ResnetBlockB_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='ResnetBlockB_Inception')

        def conv2d_(self,
                      filters,
                      num_row,
                      num_col,
                      padding='same',
                      strides=(1, 1),
                      name=None):
            return Conv2D(filters,
                        (num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn(self):
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act(self):
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(self,pooling,stride_pool):
            return  Pool(pooling,stride_pool)

    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self.conv2d_(192, 1, 1)(input_tensor)
        branch1 = self._act2b(self.batch_bn(training=training)(branch1))

        branch2 = self.conv2d_(128, 1, 1)(input_tensor)
        branch2 = self._act2b(self.batch_bn(training=training)(branch2))
        branch2 = self.conv2d_(160, 1, 7)(branch2)
        branch2 = self._act2b(self.batch_bn(training=training)(branch2))
        branch2 = self.conv2d_(192, 7, 1)(branch2)
        branch2 = self._act2b(self.batch_bn(training=training)(branch2))

        mix = concatenate([branch1, branch2], axis=3)

        final = self.conv2d_(384, 1, 1)(mix)
        final = self._act2b(self.batch_bn(training=training)(final))
        return final

class ResnetBlockC_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='ResnetBlockC_Inception')

        def conv2d_(self,
                      filters,
                      num_row,
                      num_col,
                      padding='same',
                      strides=(1, 1),
                      name=None):
            return Conv2D(filters,
                        (num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn(self):
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act(self):
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(self,pooling,stride_pool):
            return  Pool(pooling,stride_pool)

    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self.conv2d_(192, 1, 1)(input_tensor)
        branch1 = self._act2b(self.batch_bn(training=training)(branch1))

        branch2 = self.conv2d_(192, 1, 1)(input_tensor)
        branch2 = self._act2b(self.batch_bn(training=training)(branch2))
        branch2 = self.conv2d_(224, 1, 3)(branch2)
        branch2 = self._act2b(self.batch_bn(training=training)(branch2))
        branch2 = self.conv2d_(256, 3, 1)(branch2)
        branch2 = self._act2b(self.batch_bn(training=training)(branch2))

        mix = concatenate([branch1, branch2], axis=3)

        final = self.conv2d_(384, 1, 1)(mix)
        final = self._act2b(self.batch_bn(training=training)(final))
        return final
