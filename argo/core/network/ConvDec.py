import tensorflow as tf
import sonnet as snt
from ..keras_models.keras_utils import get_renorm_clipping
from ..utils.argo_utils import make_list

class ConvDec(snt.AbstractModule):

    def __init__(self,
                 channels,
                 kernel_shape,
                 # padding="SAME",
                 activation,
                 is_training,
                 final_activation=True,
                 linear_first = None, # {"sizes" : [128 * 7 * 7, ...], "reshape" : (7, 7, 128)]},
                 name="conv_dec",
                 prob_drop=0.1,
                 bn_momentum=0.99,
                 bn_renormalization=True,
                 output_shape = None,
                 **extra_params):

        """
        Args:
            output_shape (tuple): output shape to enforce (optionoal). In case it is specified, last layer will have this output shape.
            name (type): module name.
            activation (type): activation used for the internal layers.
            **extra_params (type): all the additional keyword arguments will be passed to the snt.Conv2D layers. (initializers, regularizers)
        """

        super().__init__(name=name)
        self._hidden_channels = channels
        self._num_layers = len(channels)
        # self._Conv_Class = snt.Conv2DTranspose if use_deconv else snt.Conv2D

        self._stride = 2
        self._kernel_shape = kernel_shape

        self._padding = snt.SAME
        # self._final_padding = padding

        self._output_shape = output_shape

        self._prob_drop = prob_drop
        self._activation = activation
        self._final_activation = final_activation
        self._extra_params = extra_params

        self._bn_renormalization = bn_renormalization
        # instantiate all the convolutional layers
        self._bn_momentum = bn_momentum

        self._renorm_clipping = None
        if self._bn_renormalization:
            self._renorm_clipping=get_renorm_clipping()

        if linear_first is not None:
            self._linear_first_sizes = make_list(linear_first["sizes"])
            self._linear_first_reshape = linear_first["reshape"]

        self._linear_first = linear_first

        self._is_training = is_training

    def _dropout(self, net, training):
        net = tf.layers.dropout(net, self._prob_drop, training=training)
        return net

    def _decide_stride(self, index):
        return self._stride if (index % 2 == 0) else 1

    def _build(self, inputs):
        """
        Args:
            inputs (type): node of input.
            is_training (type): tells to batchnorm if to generate the update ops.

        Returns:
            logits

        """

        net = inputs

        #LINEAR BLOCK WITH RESHAPE IF NEEDED
        # if linear_first I add extra Linear layers
        if self._linear_first is not None:
            self.linear_layers = [snt.Linear(
                                        name="linear_{}".format(i),
                                        output_size=self._linear_first_sizes[i],
                                        use_bias=True,
                                        **self._extra_params
                                        )
                                        for i in range(len(self._linear_first_sizes))]



            for i, layer in enumerate(self.linear_layers):
                net = layer(net)
                net = self._dropout(net, training=self._is_training)
                net = tf.layers.batch_normalization(net, training=self._is_training,
                                                    momentum=self._bn_momentum,
                                                    renorm=self._bn_renormalization,
                                                    renorm_momentum=self._bn_momentum,
                                                    renorm_clipping=self._renorm_clipping,
                                                    name="batch_norm_lin_{}".format(i))
                net = self._activation(net)

            net = snt.BatchReshape(shape = self._linear_first_reshape)(net)

        #CONV BLOCKS FROM HERE
        self.layers = [snt.Conv2DTranspose(
                                name="conv_2d_T_{}".format(i),
                                output_channels=self._hidden_channels[i],
                                kernel_shape=self._kernel_shape,
                                stride = self._decide_stride(i),
                                padding=self._padding,
                                use_bias=True,
                                **self._extra_params
                                )
                        for i in range(self._num_layers-1)]

        li = self._num_layers-1

        if self._output_shape is None:
            lastlayer = snt.Conv2DTranspose(
                        name="conv_2d_T_{}".format(li),
                        output_channels=self._hidden_channels[li],
                        kernel_shape=self._kernel_shape,
                        stride=self._decide_stride(li),
                        padding=self._padding,
                        use_bias=True,
                        **self._extra_params
                    )
        else:
            lastlayer = snt.Conv2DTranspose(
                name="conv_2d_T_{}".format(li),
                output_channels=self._hidden_channels[li],
                kernel_shape=self._kernel_shape,
                output_shape=self._output_shape,
                use_bias=True,
                **self._extra_params
            )

        self.layers.append(lastlayer)

        # connect them to the graph, adding batch norm and non-linearity
        for i, layer in enumerate(self.layers):
            net = layer(net)
            net = self._dropout(net, training=self._is_training)
            net = tf.layers.batch_normalization(net, training=self._is_training,
                                                momentum=self._bn_momentum,
                                                renorm=self._bn_renormalization,
                                                renorm_momentum=self._bn_momentum,
                                                renorm_clipping=self._renorm_clipping,
                                                name="batch_norm_{}".format(i))

            # no activation at the end
            if i < li:
                net = self._activation(net)

        if self._final_activation:
            net = self._activation(net)

        return net
