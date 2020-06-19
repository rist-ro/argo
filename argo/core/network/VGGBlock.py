import tensorflow as tf
import sonnet as snt
from ..keras_models.keras_utils import get_renorm_clipping
from ..utils.argo_utils import make_list
from .Identity import Identity    

class VGGBlock(snt.AbstractModule):

    def __init__(self,
                 channels,
                 is_training,
                 kernel_shape=[3, 3],
                 padding="VALID",
                 linear_last=None,
                 name="vggB",
                 activation=tf.nn.relu,
                 prob_drop=0.1,
                 bn_momentum=0.99,
                 bn_renormalization=True,
                 features_name=None,
                 **extra_params):

        """

        Args:
            num_outputs (type): number of outputs of the module.
            name (type): module name.
            activation (type): activation used for the internal layers.
            **extra_params (type): alls the additional keyword arguments will be passed to the snt.Conv2D layers. (initializers, regularizers)

        """

        super().__init__(name=name)
        self._hidden_channels = channels

        if linear_last is not None:
            linear_last = make_list(linear_last)

        self._linear_last = linear_last

        self._stride = 2
        self._kernel_shape = kernel_shape

        self._middle_padding = snt.SAME
        self._final_padding = padding

        self._prob_drop = prob_drop
        self._activation = activation
        self._extra_params = extra_params

        self._bn_renormalization = bn_renormalization

        self._is_training = is_training

        self._bn_momentum = bn_momentum

        self._features_name = features_name

        self._renorm_clipping = None
        if self._bn_renormalization:
            self._renorm_clipping=get_renorm_clipping()

    def _dropout(self, net, training):
        net = tf.layers.dropout(net, self._prob_drop, training=training)
        return net

    def _build(self, inputs):
        """
        Args:
            inputs (type): node of input.
            is_training (type): tells to batchnorm if to generate the update ops.

        """

        net = inputs

        self.layer_one = snt.Conv2D(name="conv_2d_1",
                             output_channels=self._hidden_channels,
                             kernel_shape=self._kernel_shape,
                             #stride=1,
                             padding=self._middle_padding,
                             use_bias=True,
                             **self._extra_params)

        self.layer_two = snt.Conv2D(name="conv_2d_2",
                             output_channels=self._hidden_channels,
                             kernel_shape=self._kernel_shape,
                             #stride=1,
                             padding=self._middle_padding,
                             use_bias=True,
                             **self._extra_params)

        # LAYER1
        net = self.layer_one(net)
        net = self._dropout(net, training=self._is_training)
        net = tf.layers.batch_normalization(net, training=self._is_training,
                                            momentum=self._bn_momentum,
                                            renorm=self._bn_renormalization,
                                            renorm_momentum=self._bn_momentum,
                                            renorm_clipping=self._renorm_clipping,
                                            name="batch_norm_1")


        net = self._activation(net)

        # LAYER2
        net = self.layer_two(net)
        net = self._dropout(net, training=self._is_training)
        net = tf.layers.batch_normalization(net, training=self._is_training,
                                            momentum=self._bn_momentum,
                                            renorm=self._bn_renormalization,
                                            renorm_momentum=self._bn_momentum,
                                            renorm_clipping=self._renorm_clipping,
                                            name="batch_norm_2")


        net = self._activation(net)

        if self._features_name != None:
            net = Identity(name=self._features_name)(net)

        net = tf.layers.max_pooling2d(net, pool_size=self._stride, strides=self._stride, padding=self._final_padding)

        #LINEAR BLOCK WITH RESHAPE IF NEEDED
        if self._linear_last is not None:
            self.linear_layers = [snt.Linear(
                                        name="linear_{}".format(i),
                                        output_size=self._linear_last[i],
                                        use_bias=True,
                                        **self._extra_params
                                        )
                                        for i in range(len(self._linear_last))]

            net = snt.BatchFlatten()(net)

            nl = len(self._linear_last)

            for i, layer in enumerate(self.linear_layers):
                net = layer(net)
                net = self._dropout(net, training=self._is_training)
                net = tf.layers.batch_normalization(net, training=self._is_training,
                                                    momentum=self._bn_momentum,
                                                    renorm=self._bn_renormalization,
                                                    renorm_momentum=self._bn_momentum,
                                                    renorm_clipping=self._renorm_clipping,
                                                    name="batch_norm_lin_{}".format(i))

                # if i < nl-1:
                net = self._activation(net)


        return net
