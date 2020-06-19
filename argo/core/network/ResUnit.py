import tensorflow as tf
import sonnet as snt
from .Conv2DWN import Conv2DWN

class ResUnit(snt.AbstractModule):
    def __init__(self, depth, name="resunit", kernel_shape=[3,3], stride=1, activation=tf.nn.relu, use_weight_norm=False, **extra_params):
        """
        Args:
            depth (int): the depth of the resUnit.
            name (str): module name.
            kernel_shape (int or [int,int]): the kernel size
            stride (int): the stride
            activation (tf function): activation used for the internal layers.
            **extra_params: all the additional keyword arguments will be passed to snt.Conv2D layers.
        """
        super().__init__(name=name)
        self._depth = depth
        self._num_layers = 2
        self._kernel_shapes = [kernel_shape]*2
        self._strides = [stride, 1]
        self._padding = snt.SAME
        self._activation = activation
        self._extra_params = extra_params
        self._downsample_input = False
        if stride!=1:
            self._downsample_input = True
        
        self._use_weight_norm = use_weight_norm

    def _build(self, inputs, is_training=True, test_local_stats=False):
        """
        Args:
            inputs (type): node of input.
            is_training (type): tells to batchnorm if to generate the update ops.
            test_local_stats (type): used to test local stats in batch norm.

        Returns:
            logits
        """

        net = inputs
        if inputs.shape[1]!=inputs.shape[2]:
            raise ValueError("ResUnit expects a square image")

        img_dim = inputs.shape[2]
        img_channels = inputs.shape[3]

        # instantiate all the convolutional layers
        self.layers = [Conv2DWN(name="conv2d_wn_{}".format(i),
                             output_channels=self._depth,
                             kernel_shape=self._kernel_shapes[i],
                             stride=self._strides[i],
                             padding=self._padding,
                             use_bias=True,
                             use_weight_norm=self._use_weight_norm,
                             **self._extra_params
                             ) for i in range(self._num_layers)]
        # connect them to the graph, adding batch norm and non-linearity
        for i, layer in enumerate(self.layers):
            bn = snt.BatchNorm(name="batchnorm_{}".format(i))
            net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
            net = self._activation(net)
            net = layer(net)

        inputstoadd = inputs
        if self._downsample_input:
            inputstoadd = snt.Conv2D(name="conv2d_downsample",
                                 output_channels=self._depth,
                                 kernel_shape=[1,1],
                                 stride=self._strides[0],
                                 padding=self._padding,
                                 use_bias=True,
                                 **self._extra_params
                                 )(inputs)

        logits = net + inputstoadd
        return logits
