import tensorflow as tf
import sonnet as snt
from .ResUnit import ResUnit
from .Conv2DWN import Conv2DWN


class ResNet18(snt.AbstractModule):

  def __init__(self, output_size, name="resnet18", use_weight_norm=False, activation=tf.nn.relu, **extra_params):
    """
    Args:
        num_outputs (int): the number of outputs of the module.
        name (str): module name.
        activation (tf function): activation used for the internal layers.
        **extra_params: all the additional keyword arguments will be passed to all the Conv2DWN and to the ResUnit layers.

    """
    super().__init__(name=name)
    self._output_size = output_size

    self._conv_channels = 64
    self._conv_kernel_shape = [7,7]
    self._conv_stride = 2

    self._pooling_kernel_shape = [2,2]
    self._pooling_stride = 2

    self._resunit_channels = [
        64, 64, 128, 128, 256, 256, 512, 512
        ]
    self._num_resunits = len(self._resunit_channels)
    # first kernel 7x7 all the rest 3x3.
    self._resunit_kernel_shapes = [[3, 3]] * self._num_resunits
    self._resunit_strides = [1, 1, 2, 1, 2, 1, 2, 1]

    self._padding = snt.SAME
    self._activation = activation
    self._use_weight_norm = use_weight_norm
    self._extra_params = extra_params


  def _build(self, inputs, is_training=True, test_local_stats=False):
    """
    Args:
        inputs (type): node of input.
        is_training (type): tells to batchnorm if to generate the update ops.
        test_local_stats (type): used to test local stats in batch norm.

    Returns:
        logits

    """
    # instantiate all the convolutional layers
    self.layers = [Conv2DWN(name="conv2d_wn",
                         output_channels=self._conv_channels,
                         kernel_shape=self._conv_kernel_shape,
                         stride=self._conv_stride,
                         padding=self._padding,
                         use_bias=True,
                         use_weight_norm=self._use_weight_norm,
                         **self._extra_params)]
#(self, depth, name="resUnit", kernel_shape=[3,3], stride=1, activation=tf.nn.relu, **extra_params)
    for i in range(self._num_resunits):
        self.layers.append(ResUnit(
                             depth= self._resunit_channels[i],
                             name="resunit_{}".format(i),
                             kernel_shape=self._resunit_kernel_shapes[i],
                             stride=self._resunit_strides[i],
                             activation=self._activation,
                             use_weight_norm=self._use_weight_norm,
                             **self._extra_params))

    net = self.layers[0](inputs)
    net = tf.layers.max_pooling2d(
                        net,
                        self._pooling_kernel_shape,
                        self._pooling_stride,
                        padding=self._padding,
                        data_format='channels_last',
                        name="max_pooling2d"
                    )

    for i, resunit in enumerate(self.layers[1:]):
        net = resunit(net)

    net = tf.layers.average_pooling2d(
                        net,
                        self._pooling_kernel_shape,
                        self._pooling_stride,
                        padding=self._padding,
                        data_format='channels_last',
                        name="avg_pooling2d"
                    )

    net = snt.BatchFlatten(name="flatten")(net)
    logits = snt.Linear(self._output_size)(net)

    return logits
