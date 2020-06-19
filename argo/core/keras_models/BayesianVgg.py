import tensorflow as tf
import tensorflow_probability as tfp
from .keras_utils import get_renorm_clipping, get_keras_activation, act_id
from .ArgoKerasModel import ArgoKerasModel
from ..utils.argo_utils import listWithPoints, make_list
from tensorflow.python.util import tf_inspect

class BayesianVgg(ArgoKerasModel):
    """Constructs a VGG16 model.

    Args:
    input_shape: A `tuple` indicating the Tensor shape.
    num_classes: `int` representing the number of class labels.
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

    """

    def _id(self):
        _id = 'BVGG'
        _id += "_f" + listWithPoints(self._filters)
        _id += "_fl" + str(int(self._flipout))
        _id += "_rn" + str(int(self._renorm))
        _id += "_a" + act_id(self._activation_tuple)
        _id += "_fa" + str(int(self._final_activation))

        pooling_dict = {
            "max": 'M',
            "avg": 'A',
            None: 'N'
        }
        _id += "_p" + pooling_dict[self._pooling]
        
        if self._linear_last is not None:
            _id += "_l" + listWithPoints(self._linear_last)
        
        return _id

    def __init__(self,
                 filters=[16, 16, 32, 32, 32],  # [32, 64, 64, 128, 128],
                 kernels=[3, 3, 3, 3, 3],
                 strides=[2, 2, 2, 2, 2],
                 linear_last=None,
                 flipout=True,
                 pooling="max",
                 renorm=False,
                 activation=('relu', {'alpha':0.3}),
                 final_activation=True,
                 layer_kwargs={},
                 layer_kwargs_bayes={}):

        super().__init__(name='bayesian_vgg')

        self._flipout = flipout
        self._filters = filters
        self._renorm = renorm
        self._pooling = pooling
        self._final_activation = final_activation

        n_blocks = len(kernels)
        end_activations = [True] * n_blocks

        if linear_last is not None:
            n_lin_layers = len(linear_last)
            linear_last = make_list(linear_last)
            lin_end_activations = [True] *n_lin_layers

            lin_end_activations[-1] = final_activation

        else:
            end_activations[-1] = final_activation

        self._linear_last = linear_last
        self._activation_tuple = activation

        pooling_choices = ["max", "avg", None]
        if pooling not in pooling_choices:
            raise ValueError("pooling must be in {:}, instead {:} found.".format(pooling_choices, pooling))

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

        for i in range(n_blocks):
            block = VggBlock(Conv2D,
                            filters[i],
                            kernels[i],
                            strides[i],
                            pooling,
                            renorm,
                            renorm_clipping,
                            activation,
                            final_activation=end_activations[i],
                            **layer_kwargs_bayes)
            self.blocks_list.append(block)

        # if logits_size is specified I add an extra Dense layer
        if self._linear_last is not None:
            self.blocks_list.append(tf.keras.layers.Flatten())
            # check that activations are put here
            for ls, act_bool in zip(self._linear_last, lin_end_activations):
                self.blocks_list.append(Dense(ls, **layer_kwargs_bayes))
                if act_bool:
                    self.blocks_list.append(tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping))
                    self.blocks_list.append(get_keras_activation(activation))

        self._layer_call_argspecs = {}
        for layer in self.blocks_list:
            self._layer_call_argspecs[layer.name] = tf_inspect.getfullargspec(layer.call).args

    def _layer_call_kwargs(self, layer, training):
        kwargs = {}
        argspec = self._layer_call_argspecs[layer.name]
        if 'training' in argspec:
            kwargs['training'] = training
    
        return kwargs

    def call(self, inputs, training=False, **extra_kwargs):
        net = inputs
        for layer in self.blocks_list:
            kwargs = self._layer_call_kwargs(layer, training)
            net = layer(net, **kwargs)

        return net



class VggBlock(tf.keras.Model):
    def __init__(self, Conv2D, filters, kernel, stride, pooling,
                 renorm, renorm_clipping, activation_tuple, final_activation, **layer_kwargs_bayes):

        super().__init__(name='vgg_block')

        self.conv2a = Conv2D(
                        filters,
                        kernel,
                        padding='same',
                        **layer_kwargs_bayes)

        self.bn2a = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)

        # self._act = tf.nn.leaky_relu
        self._act2a = get_keras_activation(activation_tuple)

        conv2b_stride = self._get_conv_stride(pooling, stride)

        self.conv2b = Conv2D(
                        filters,
                        kernel,
                        strides=conv2b_stride,
                        padding='same',
                        **layer_kwargs_bayes)

        self._act2b = None
        if final_activation:
            self.bn2b = tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)
            self._act2b = get_keras_activation(activation_tuple)

        self._pooling = None
        if pooling == "max":
            self._pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=stride)
        elif pooling == "avg":
            self._pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=stride)

    def _get_conv_stride(self, pooling, stride):

        if pooling is None:
            conv_stride = stride
        else:
            conv_stride = 1

        return conv_stride

    def call(self, inputs, training=None, mask=None):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = self._act2a(x)

        x = self.conv2b(x)

        if self._act2b:
            x = self.bn2b(x, training=training)
            x = self._act2b(x)

        if self._pooling is not None:
            x = self._pooling(x)

        return x
