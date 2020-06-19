import tensorflow as tf
import tensorflow_probability as tfp
from .keras_utils import get_renorm_clipping
from tensorflow.keras.layers import Concatenate
#from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D
class BayesianInception_Basic_residual(tf.keras.Model):
    """Constructs a ResNet model.
    """
    def __init__(self,
                 filters_1x1=[16, 16, 16, 32],
                 filters_3x3_reduce=[32, 32, 32, 32],
                 filters_3x3=[64, 64, 64, 64],
                 filters_5x5_reduce=[16, 16, 16, 32],
                 filters_5x5=[32, 32, 32, 32],
                 filters_pool_proj=[32, 32, 32, 32],
                 strides=[2, 2, 2, 2],
                 logits_size=None,
                 flipout=True,
                 renorm=False,
                 pooling="avg",
                 activation = ('relu', {}),
                 layer_kwargs={},
                 layer_kwargs_bayes={}):

        super().__init__(name='bayesian_inception_basic_res')
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
        for i in range(len(filters_1x1)):
            block = Module_Inception(Conv2D,
                            filters_1x1[i],
                            filters_3x3_reduce[i],
                            filters_3x3[i],
                            filters_5x5_reduce[i],
                            filters_5x5[i],
                            filters_pool_proj[i],
                            strides[i],
                            renorm,
                            Pool,
                            renorm_clipping,
                            activation_name = activation_name,
                            activation_kwargs = activation_kwargs,
                            **layer_kwargs_bayes)
            self.blocks_list.append(block)

        self._flat = tf.keras.layers.Flatten()
#        self._drop=tf.keras.layers.SpatialDropout2D(rate=0.5)
        self._last_dense = None
        if logits_size is not None:
            self._last_dense = Dense(
                                logits_size,
                                **layer_kwargs_bayes)

    def call(self, inputs, training=False, **extra_kwargs):
        net = inputs
        for block in self.blocks_list:
            net = block(net, training=training)
#        net = self._pooling(net)
#        net = self._drop(net,training=training)
        net = self._flat(net)
        if self._last_dense:
            net = self._last_dense(net)
        return net



class Module_Inception(tf.keras.Model):
    def __init__(self, Conv2D,
                 filters_1x1,
                 filters_3x3_reduce,
                 filters_3x3,
                 filters_5x5_reduce,
                 filters_5x5,
                 filters_pool_proj,
                 strides,renorm, Pool,renorm_clipping,
                 activation_name='relu', activation_kwargs={},
                 final_activation=True,
                 **layer_kwargs_bayes
                 ):

        super().__init__(name='Module_Inception')

        def conv2d_(  filters=10,
                      num_row=1,
                      num_col=1,
                      padding='same'):
            return Conv2D(filters,
                        kernel_size=(num_row, num_col),
                        padding=padding,
                        **layer_kwargs_bayes)
        def concatenate():
            return Concatenate(axis=3)

        def batch_bn():
            return tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
        def act():
            return tf.keras.layers.Activation(activation_name, **activation_kwargs)

        def pool_apply(pooling,stride,padding='same'):
            return  Pool(pooling,stride,padding=padding)

        self._conv1a =conv2d_(filters=filters_1x1, num_row=1, num_col=1,padding='same')
        self._batch1a= batch_bn()
        self._act1a= act()
        self._conv2a =conv2d_(filters=filters_3x3_reduce, num_row=1, num_col=1,padding='same')
        self._batch2a= batch_bn()
        self._act2a= act()
        self._conv2b =conv2d_(filters=filters_3x3, num_row=3, num_col=3,padding='same')
        self._batch2b= batch_bn()
        self._act2b= act()
        self._conv3a =conv2d_(filters=filters_5x5_reduce, num_row=1, num_col=1,padding='same')
        self._batch3a= batch_bn()
        self._act3a= act()
        self._conv3b =conv2d_(filters=filters_5x5, num_row=5, num_col=5,padding='same')
        self._batch3b= batch_bn()
        self._act3b= act()
        self._pool_4a=pool_apply(2,1,padding='same')
        self._conv4a =conv2d_(filters=filters_pool_proj, num_row=1, num_col=1,padding='same')
        self._batch4a= batch_bn()
        self._act4a= act()
        self._concatenate_a=concatenate()
        self._pool_5a=pool_apply(2,strides,padding='same')

    def call(self, input_tensor, training=False, **extra_kwargs):

        branch1 = self._conv1a(input_tensor)
        branch1 = self._act1a(branch1)#self._batch1a(branch1, training=training))

        branch2 = self._conv2a(input_tensor)
        branch2 = self._act2a(branch2)#self._batch2a(branch2,training=training))
        branch2 = self._conv2b(branch2)
        branch2 = self._act2b(branch2)#self._batch2b(branch2,training=training))


        branch3 = self._conv3a(input_tensor)
        branch3 = self._act3a(branch3)#self._batch3a(branch3,training=training))
        branch3 = self._conv3b(branch3)
        branch3 = self._act3b(branch3)#self._batch3b(branch3,training=training))

        branch4 = self._pool_4a(input_tensor)
        branch4 = self._conv4a(branch4)
        branch4 = self._act4a(branch4)#self._batch4a(branch4,training=training))


        branch_T = self._concatenate_a([branch1, branch2, branch3,branch4])
        branch_T = self._batch1a(branch_T,training=training)
        branch_T = self._pool_5a(branch_T)
        return branch_T
