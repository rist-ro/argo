import tensorflow as tf
import tensorflow_probability as tfp
from .keras_utils import get_renorm_clipping
from tensorflow.keras.layers import Concatenate
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

        super().__init__(name='bayesian_inception')
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


        self.blocks_list = []


        stem=Resnet_Stem_Inception(Conv2D,
                        renorm,
                        Pool,
                        renorm_clipping,
                        activation_name = activation_name,
                        activation_kwargs = activation_kwargs,
                        **layer_kwargs_bayes)

        self.blocks_list.append(stem)

        for i in range(4):
            block = ResnetBlockA_Inception(Conv2D,
                            renorm,
                            Pool,
                            renorm_clipping,
                            activation_name = activation_name,
                            activation_kwargs = activation_kwargs,
                            **layer_kwargs_bayes)
            self.blocks_list.append(block)

        redA=ReductionA_Inception(Conv2D,
                        renorm,
                        Pool,
                        renorm_clipping,
                        activation_name = activation_name,
                        activation_kwargs = activation_kwargs,
                        **layer_kwargs_bayes)

        self.blocks_list.append(redA)
        for i in range(6):
            block1 = ResnetBlockB_Inception(Conv2D,
                            renorm,
                            Pool,
                            renorm_clipping,
                            activation_name = activation_name,
                            activation_kwargs = activation_kwargs,
                            **layer_kwargs_bayes)
            self.blocks_list.append(block1)

        redB=ReductionB_Inception(Conv2D,
                        renorm,
                        Pool,
                        renorm_clipping,
                        activation_name = activation_name,
                        activation_kwargs = activation_kwargs,
                        **layer_kwargs_bayes)

        self.blocks_list.append(redB)

        for i in range(4):
            block2 = ResnetBlockC_Inception(Conv2D,
                            renorm,
                            Pool,
                            renorm_clipping,
                            activation_name = activation_name,
                            activation_kwargs = activation_kwargs,
                            **layer_kwargs_bayes)
            self.blocks_list.append(block2)

        self._pooling = tf.keras.layers.AveragePooling2D(2,1)
        self._flat = tf.keras.layers.Flatten()
        self._drop=tf.keras.layers.SpatialDropout2D(rate=0.5)
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
        net = self._drop(net,training=training)
        net = self._flat(net)

        if self._last_dense:
            net = self._last_dense(net)

        return net


class ResnetBlockA_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, Pool,renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='ResnetBlockA_Inception')

        def conv2d_(  filters=10,
                      num_row=1,
                      num_col=1,
                      padding='same',
                      strides=1,
                      name=None):
            return Conv2D(filters,
                        kernel_size=(num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn():
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act():
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(pooling,stride_pool):
            return  Pool(pooling,stride_pool)
        def concatenate():
            return Concatenate(axis=3)

        self._conv2a =conv2d_(filters=16, num_row=1, num_col=1)
        self._batch2a= batch_bn()
        self._act2a= act()
        self._conv2b =conv2d_(filters=16, num_row=1, num_col=1)
        self._batch2b= batch_bn()
        self._act2b= act()
        self._conv2c =conv2d_(filters=16, num_row=3, num_col=3)
        self._batch2c= batch_bn()
        self._act2c= act()
        self._conv2d =conv2d_(filters=16, num_row=1, num_col=1)
        self._batch2d= batch_bn()
        self._act2d= act()
        self._conv2e =conv2d_(filters=28, num_row=3, num_col=3)
        self._batch2e= batch_bn()
        self._act2e= act()
        self._conv2f =conv2d_(filters=32, num_row=3, num_col=3)
        self._batch2f= batch_bn()
        self._act2f= act()
        self._conv2g =conv2d_(filters=32, num_row=1, num_col=1)
        self._batch2g= batch_bn()
        self._act2g= act()
        self._concatenate_a=concatenate()

    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self._conv2a(input_tensor)
        branch1 = self._act2a(self._batch2a(branch1, training=training))

        branch2 = self._conv2b(input_tensor)
        branch2 = self._act2b(self._batch2b(branch2,training=training))
        branch2 = self._conv2c(branch2)
        branch2 = self._act2c(self._batch2c(branch2,training=training))

        branch3 = self._conv2d(input_tensor)
        branch3 = self._act2d(self._batch2d(branch3,training=training))
        branch3 = self._conv2e(branch3)
        branch3 = self._act2e(self._batch2e(branch3,training=training))
        branch3 = self._conv2f(branch3)
        branch3 = self._act2f(self._batch2f(branch3,training=training))

        mix = self._concatenate_a([branch1, branch2,branch3])
        final = self._conv2g(mix)
        final = self._act2g(self._batch2g(final,training=training))
        return final
class ResnetBlockB_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, Pool,renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='ResnetBlockB_Inception')

        def conv2d_(  filters=10,
                      num_row=1,
                      num_col=1,
                      padding='same',
                      strides=1,
                      name=None):
            return Conv2D(filters,
                        kernel_size=(num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn():
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act():
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(pooling,stride_pool):
            return  Pool(pooling,stride_pool)
        def concatenate():
            return Concatenate(axis=3)

        self._conv2a =conv2d_(filters=32, num_row=1, num_col=1)
        self._batch2a= batch_bn()
        self._act2a= act()
        self._conv2b =conv2d_(filters=32, num_row=1, num_col=1)
        self._batch2b= batch_bn()
        self._act2b= act()
        self._conv2c =conv2d_(filters=32, num_row=1, num_col=7)
        self._batch2c= batch_bn()
        self._act2c= act()
        self._conv2d =conv2d_(filters=32, num_row=7, num_col=1)
        self._batch2d= batch_bn()
        self._act2d= act()
        self._conv2e =conv2d_(filters=32, num_row=1, num_col=1)
        self._batch2e= batch_bn()
        self._act2e= act()
        self._concatenate_a=concatenate()
    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self._conv2a(input_tensor)
        branch1 = self._act2a(self._batch2a(branch1, training=training))

        branch2 = self._conv2b(input_tensor)
        branch2 = self._act2b(self._batch2b(branch2,training=training))
        branch2 = self._conv2c(branch2)
        branch2 = self._act2c(self._batch2c(branch2,training=training))
        branch2 = self._conv2d(branch2)
        branch2 = self._act2d(self._batch2d(branch2,training=training))


        mix = self._concatenate_a([branch1, branch2])
        final = self._conv2e(mix)
        final = self._act2e(self._batch2e(final,training=training))
        return final
class ResnetBlockC_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, Pool,renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='ResnetBlockC_Inception')

        def conv2d_(  filters=10,
                      num_row=1,
                      num_col=1,
                      padding='same',
                      strides=1,
                      name=None):
            return Conv2D(filters,
                        kernel_size=(num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn():
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act():
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(self,pooling,stride_pool):
            return  Pool(pooling,stride_pool)
        def concatenate():
            return Concatenate(axis=3)

        self._conv2a =conv2d_(filters=16, num_row=1, num_col=1)
        self._batch2a= batch_bn()
        self._act2a= act()
        self._conv2b =conv2d_(filters=16, num_row=1, num_col=1)
        self._batch2b= batch_bn()
        self._act2b= act()
        self._conv2c =conv2d_(filters=32, num_row=1, num_col=3)
        self._batch2c= batch_bn()
        self._act2c= act()
        self._conv2d =conv2d_(filters=32, num_row=3, num_col=1)
        self._batch2d= batch_bn()
        self._act2d= act()
        self._conv2e =conv2d_(filters=32, num_row=1, num_col=1)
        self._batch2e= batch_bn()
        self._act2e= act()
        self._concatenate_a=concatenate()
    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self._conv2a(input_tensor)
        branch1 = self._act2a(self._batch2a(branch1, training=training))

        branch2 = self._conv2b(input_tensor)
        branch2 = self._act2b(self._batch2b(branch2,training=training))
        branch2 = self._conv2c(branch2)
        branch2 = self._act2c(self._batch2c(branch2,training=training))
        branch2 = self._conv2d(branch2)
        branch2 = self._act2d(self._batch2d(branch2,training=training))


        mix = self._concatenate_a([branch1, branch2])
        final = self._conv2e(mix)
        final = self._act2e(self._batch2e(final,training=training))
        return final

class Resnet_Stem_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, Pool,renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='Stem_Inception')

        def conv2d_(  filters=10,
                      num_row=1,
                      num_col=1,
                      padding='same',
                      strides=1,
                      name=None):
            return Conv2D(filters,
                        kernel_size=(num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)
        def concatenate():
            return Concatenate(axis=3)

        def batch_bn():
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act():
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(pooling,stride_pool,padding='valid'):
            return  Pool(pooling,stride_pool,padding=padding)

        self._conv2a =conv2d_(filters=16, num_row=3, num_col=3,strides=2,padding='valid')
        self._batch2a= batch_bn()
        self._act2a= act()
        self._conv2b =conv2d_(filters=16, num_row=3, num_col=3,padding='valid')
        self._batch2b= batch_bn()
        self._act2b= act()
        self._conv2c =conv2d_(filters=16, num_row=3, num_col=3,padding='valid')
        self._batch2c= batch_bn()
        self._act2c= act()
        self._pool_2c=pool_apply(3,1)
        self._conv2d =conv2d_(filters=16, num_row=3, num_col=3,padding='valid')
        self._batch2d= batch_bn()
        self._act2d= act()

        self._conv2e =conv2d_(filters=32, num_row=1, num_col=1)
        self._batch2e= batch_bn()
        self._act2e= act()
        self._conv2e1 =conv2d_(filters=32, num_row=7, num_col=1)
        self._batch2e1= batch_bn()
        self._act2e1= act()
        self._conv2e2 =conv2d_(filters=32, num_row=1, num_col=7)
        self._batch2e2= batch_bn()
        self._act2e2= act()
        self._conv2e3 =conv2d_(filters=32, num_row=3, num_col=3, padding='valid')
        self._batch2e3= batch_bn()
        self._act2e3= act()

        self._conv2f =conv2d_(filters=32, num_row=1, num_col=1, padding='valid')
        self._batch2f= batch_bn()
        self._act2f= act()
        self._conv2f1 =conv2d_(filters=32, num_row=3, num_col=3, padding='valid')
        self._batch2f1= batch_bn()
        self._act2f1= act()

        self._conv2g =conv2d_(filters=32, num_row=3, num_col=3,padding='valid')
        self._batch2g= batch_bn()
        self._act2g= act()
        self._pool_2g=pool_apply(3,1)
        self._concatenate_a=concatenate()
        self._concatenate_b=concatenate()
        self._concatenate_c=concatenate()

    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self._conv2a(input_tensor)
        branch1 = self._act2a(self._batch2a(branch1, training=training))
        branch1 = self._conv2b(branch1)
        branch1 = self._act2b(self._batch2b(branch1,training=training))
        branch1 = self._conv2c(branch1)
        branch1 = self._act2c(self._batch2c(branch1,training=training))

        branch2 = self._pool_2c(branch1)
        branch3 = self._conv2d(branch1)
        branch3 = self._act2d(self._batch2d(branch3,training=training))

        mix = self._concatenate_a([branch3, branch2])
        branch1 = self._conv2e(mix)
        branch1 = self._act2e(self._batch2e(branch1,training=training))
        branch1 = self._conv2e1(branch1)
        branch1 = self._act2e1(self._batch2e1(branch1,training=training))
        branch1 = self._conv2e2(branch1)
        branch1 = self._act2e2(self._batch2e2(branch1,training=training))
        branch1 = self._conv2e3(branch1)
        branch1 = self._act2e3(self._batch2e3(branch1,training=training))
        branch2 = self._conv2f(mix)
        branch2 = self._act2f(self._batch2f(branch2,training=training))
        branch2 = self._conv2f1(branch2)
        branch2 = self._act2f1(self._batch2f1(branch2,training=training))
        mix = self._concatenate_b([branch1, branch2])
        branch1 = self._conv2g(mix)
        branch1 = self._act2g(self._batch2g(branch1,training=training))
        branch2 = self._pool_2g(mix)
        branch = self._concatenate_c([branch1, branch2])

        return branch
# class Resnet_Stem_test(tf.keras.Model):
#     def __init__(self, Conv2D,
#                  renorm, Pool,renorm_clipping,
#                  activation_name='relu', activation_kwargs={},
#                  final_activation=True,
#                  **layer_kwargs_bayes
#                  ):
#
#         super().__init__(name='Stem_Inception')
#
#         def conv2d_(  filters=10,
#                       num_row=1,
#                       num_col=1,
#                       padding='same',
#                       strides=1,
#                       name=None):
#             return Conv2D(filters,
#                         kernel_size=(num_row, num_col),
#                         strides=strides,
#                         padding=padding,
#                         **layer_kwargs_bayes)
#         def concatenate():
#             return Concatenate(axis=3)
#
#         def batch_bn():
#             return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
#         def act():
#             return tf.keras.layers.Activation(activation_name, **activation_kwargs)
#
#         def pool_apply(pooling,stride_pool,padding='valid'):
#             return  Pool(pooling,stride_pool,padding=padding)
#
#         self._conv2a =conv2d_(filters=32, num_row=3, num_col=3,strides=2,padding='valid')
#         self._batch2a= batch_bn()
#         self._act2a= act()
#         self._conv2b =conv2d_(filters=32, num_row=3, num_col=3,padding='valid')
#         self._batch2b= batch_bn()
#         self._act2b= act()
#         self._conv2c =conv2d_(filters=64, num_row=3, num_col=3,padding='valid')
#         self._batch2c= batch_bn()
#         self._act2c= act()
#         self._pool_2c=pool_apply(3,1)
#         self._conv2d =conv2d_(filters=64, num_row=3, num_col=3,padding='valid')
#         self._batch2d= batch_bn()
#         self._act2d= act()
#
#
#         self._concatenate_a=concatenate()
#         self._concatenate_b=concatenate()
#         self._concatenate_c=concatenate()
#
#     def call(self, input_tensor, training=False, **extra_kwargs):
#
#         branch1 = self._conv2a(input_tensor)
#         branch1 = self._act2a(self._batch2a(branch1, training=training))
#         branch1 = self._conv2b(branch1)
#         branch1 = self._act2b(self._batch2b(branch1,training=training))
#         branch1 = self._conv2c(branch1)
#         branch1 = self._act2c(self._batch2c(branch1,training=training))
#
#         branch2 = self._pool_2c(branch1)
#         branch3 = self._conv2d(branch1)
#         branch3 = self._act2d(self._batch2d(branch3,training=training))
#
#         mix = self._concatenate_a([branch3, branch2])
#
#
#         return mix

class ReductionA_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, Pool,renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='ReductionA_Inception')

        def conv2d_(  filters=10,
                      num_row=1,
                      num_col=1,
                      padding='same',
                      strides=1,
                      name=None):
            return Conv2D(filters,
                        kernel_size=(num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn():
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act():
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(pooling,stride_pool):
            return  Pool(pooling,stride_pool)
        def concatenate():
            return Concatenate(axis=3)
        self._pool_2a=pool_apply(3,2)
        self._conv2a =conv2d_(filters=16, num_row=3, num_col=3,strides=2,padding='valid')
        self._batch2a= batch_bn()
        self._act2a= act()
        self._conv2b =conv2d_(filters=16, num_row=1, num_col=1)
        self._batch2b= batch_bn()
        self._act2b= act()
        self._conv2c =conv2d_(filters=16, num_row=3, num_col=3)
        self._batch2c= batch_bn()
        self._act2c= act()
        self._conv2d =conv2d_(filters=16, num_row=3, num_col=3,strides=2,padding='valid')
        self._batch2d= batch_bn()
        self._act2d= act()
        self._concatenate_a=concatenate()

    def call(self, input_tensor, training=False, **extra_kwargs):
        branch0 = self._pool_2a(input_tensor)
        branch1 = self._conv2a(input_tensor)
        branch1 = self._act2a(self._batch2a(branch1, training=training))
        branch2 = self._conv2b(input_tensor)
        branch2 = self._act2b(self._batch2b(branch2,training=training))
        branch2 = self._conv2c(branch2)
        branch2 = self._act2c(self._batch2c(branch2,training=training))
        branch2 = self._conv2d(branch2)
        branch2 = self._act2d(self._batch2d(branch2,training=training))

        final = self._concatenate_a([branch1, branch2,branch0])
        return final

class ReductionB_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 renorm, Pool,renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='ReductionB_Inception')

        def conv2d_(  filters=10,
                      num_row=1,
                      num_col=1,
                      padding='same',
                      strides=1,
                      name=None):
            return Conv2D(filters,
                        kernel_size=(num_row, num_col),
                        strides=strides,
                        padding=padding,
                        **layer_kwargs_bayes)

        def batch_bn():
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act():
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(pooling,stride_pool):
            return  Pool(pooling,stride_pool)
        def concatenate():
            return Concatenate(axis=3)



        self._pool_2a=pool_apply(3,2)
        self._conv2a =conv2d_(filters=16, num_row=1, num_col=1)
        self._batch2a= batch_bn()
        self._act2a= act()
        self._conv2b =conv2d_(filters=16, num_row=3, num_col=3,strides=2,padding='valid')
        self._batch2b= batch_bn()
        self._act2b= act()
        self._conv2c =conv2d_(filters=16, num_row=1, num_col=1)
        self._batch2c= batch_bn()
        self._act2c= act()
        self._conv2d =conv2d_(filters=16, num_row=3, num_col=3,strides=2,padding='valid')
        self._batch2d= batch_bn()
        self._act2d= act()

        self._conv2e =conv2d_(filters=32, num_row=1, num_col=1)
        self._batch2e= batch_bn()
        self._act2e= act()

        self._conv2f =conv2d_(filters=32, num_row=3, num_col=3)
        self._batch2f= batch_bn()
        self._act2f= act()
        self._conv2g =conv2d_(filters=32, num_row=3, num_col=3,strides=2,padding='valid')
        self._batch2g= batch_bn()
        self._act2g= act()
        self._concatenate_a=concatenate()

    def call(self, input_tensor, training=False, **extra_kwargs):
        branch2 = self._pool_2a(input_tensor)

        branch1 = self._conv2a(input_tensor)
        branch1 = self._act2a(self._batch2a(branch1, training=training))
        branch1 = self._conv2b(branch1)
        branch1 = self._act2b(self._batch2b(branch1,training=training))

        branch3 = self._conv2c(input_tensor)
        branch3 = self._act2c(self._batch2c(branch3,training=training))
        branch3 = self._conv2d(branch3)
        branch3 = self._act2d(self._batch2d(branch3,training=training))

        branch4 = self._conv2e(input_tensor)
        branch4 = self._act2e(self._batch2e(branch4,training=training))
        branch4 = self._conv2f(branch4)
        branch4 = self._act2f(self._batch2f(branch4,training=training))
        branch4 = self._conv2g(branch4)
        branch4 = self._act2g(self._batch2g(branch4,training=training))

        final = self._concatenate_a([branch1, branch3,branch2,branch4])
        return final
