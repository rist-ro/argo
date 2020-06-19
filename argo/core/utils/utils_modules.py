import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp
import importlib
from ..keras_models.keras_utils import get_renorm_clipping

def network_layer_builder(module_name, module_kwargs):
    # load the module specified
    pymodule = importlib.import_module("..network." + module_name,
                                             '.'.join(__name__.split('.')[:-1]))

    module_class = getattr(pymodule, module_name)

    module = module_class(**module_kwargs)

    return module


class Vgg_Bernoulli(snt.AbstractModule):

    def __init__(self, output_size,
                 filters=[16, 16, 32, 32, 32],  # [32, 64, 64, 128, 128],
                 kernels=[3, 3, 3, 3, 3],
                 strides=[2, 2, 2, 2, 2],
                 name="vggB",
                 activation=tf.nn.relu,
                 drop_connect=False,
                 prob_drop=0.1,
                 bn_momentum=0.99,
                 bn_renormalization=True,
                 aleatoric_layer = ("MultivariateNormalDiag", {}),
                 **extra_params):
        """

        Args:
            num_outputs (type): number of outputs of the module.
            name (type): module name.
            activation (type): activation used for the internal layers.
            **extra_params (type): alls the additional keyword arguments will be passed to the snt.Conv2D layers. (initializers, regularizers)

        """
        super().__init__(name=name)
        self._output_size = output_size
        self._hidden_channels = filters
        self._strides = strides
        self._kernel_shapes = kernels  # All kernels are 3x3.
        self._num_layers = len(self._hidden_channels)
        self._paddings = [snt.SAME] * self._num_layers

        self._prob_drop = prob_drop
        self._drop_connect = drop_connect
        self._activation = activation
        self._extra_params = extra_params
        self._aleatoric_layer_tuple = aleatoric_layer
        self._bn_renormalization = bn_renormalization
        # instantiate all the convolutional layers

        def custom_getter_dropconnect(getter, name, *args, **kwargs):
            shape = kwargs["shape"]
            theta = getter(name, *args, **kwargs)

            theta_masked = tf.layers.dropout(theta, self._prob_drop, training=True)

            return theta_masked

        # custom_getter = custom_getter_dropconnect if self._drop_connect else None
        self._custom_getter = {"w" : custom_getter_dropconnect} if self._drop_connect else None

        self._bn_momentum = bn_momentum

        self._renorm_clipping = None
        if self._bn_renormalization:
            self._renorm_clipping=get_renorm_clipping()

    def maybe_dropout(self, net):
        if self._drop_connect == False:
            net = tf.layers.dropout(net, self._prob_drop, training=True)
        return net

    def _build(self, inputs, is_training=True, test_local_stats=True):
        """
        Args:
            inputs (type): node of input.
            is_training (type): tells to batchnorm if to generate the update ops.
            test_local_stats (type): used to test local stats in batch norm.

        Returns:
            logits

        """

        net = inputs

        # self._custom_getter_layer = [self._custom_getter]*(self._num_layers)

        self.layers_one = [snt.Conv2D(name="conv_2d_1_{}".format(i),
                             output_channels=self._hidden_channels[i],
                             kernel_shape=self._kernel_shapes[i],
                             #stride=self._strides[i],
                             padding=self._paddings[i],
                             use_bias=True,
                             custom_getter= self._custom_getter,
                             **self._extra_params
                             ) for i in range(self._num_layers)]

        self.layers_two = [snt.Conv2D(name="conv_2d_2_{}".format(i),
                             output_channels=self._hidden_channels[i],
                             kernel_shape=self._kernel_shapes[i],
                            # stride=self._strides[i],
                             padding=self._paddings[i],
                             use_bias=True,
                              custom_getter= self._custom_getter,
                             **self._extra_params
                             ) for i in range(self._num_layers)]

        # connect them to the graph, adding batch norm and non-linearity
        for i, (layer_one, layer_two) in enumerate(zip(self.layers_one, self.layers_two)):

            # LAYER1
            net = layer_one(net)
            # bn = snt.BatchNorm(name="batch_norm_1_{}".format(i))
            # net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
            net = self.maybe_dropout(net)
            net = tf.layers.batch_normalization(net, training=is_training,
                                                momentum=self._bn_momentum,
                                                renorm=self._bn_renormalization,
                                                renorm_momentum=self._bn_momentum,
                                                renorm_clipping=self._renorm_clipping,
                                                name="batch_norm_1_{}".format(i))


            net = self._activation(net)

            # LAYER1

            # LAYER2
            net = layer_two(net)
            # bn = snt.BatchNorm(name="batch_norm_2_{}".format(i))
            # net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
            net = self.maybe_dropout(net)
            net = tf.layers.batch_normalization(net, training=is_training,
                                                momentum=self._bn_momentum,
                                                renorm=self._bn_renormalization,
                                                renorm_momentum=self._bn_momentum,
                                                renorm_clipping=self._renorm_clipping,
                                                name="batch_norm_2_{}".format(i))


            net = self._activation(net)
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=self._strides[i], padding="VALID")
            # LAYER2

        net = snt.BatchFlatten(name="flatten")(net)

        # net = self.maybe_dropout(net)

        # if  self._drop_connect == False:
        #     net= tf.layers.dropout(net,self._prob_drop, training=True)

        name, kwargs = self._aleatoric_layer_tuple
        aleatoric_kwargs = {
            **self._extra_params,
            "output_size": self._output_size,

            "custom_getter": None # self._custom_getter

            # no custom getter on the last layer
            # "custom_getter": self._custom_getter

        }

        self._distr_model = network_layer_builder(name, aleatoric_kwargs)

        distr = self._distr_model(net)

        return distr

class Alex_Bernoulli(snt.AbstractModule):

    def __init__(self, output_size,
                 filters=[32, 32, 32, 32, 32],#[64, 192,192,256,256],  # [32, 64, 64, 128, 128],
                 kernels=[7,5,3,3,3],
                 strides=[4, 2, 2, 2, 2],
                 #strides=[2, 2, 2, 2, 2],
                 name="AlexB",
                 activation=tf.nn.relu,
                 drop_connect=False,
                 prob_drop=0.1,
                 bn_renormalization=False,
                 bn_momentum=0.99,
                 aleatoric_layer = ("MultivariateNormalDiag", {}),
                 **extra_params):
        """

        Args:
            num_outputs (type): number of outputs of the module.
            name (type): module name.
            activation (type): activation used for the internal layers.
            **extra_params (type): alls the additional keyword arguments will be passed to the snt.Conv2D layers. (initializers, regularizers)

        """
        super().__init__(name=name)
        self._output_size = output_size
        self._hidden_channels = filters
        self._strides = strides
        self._kernel_shapes = kernels  # All kernels are 3x3.
        self._num_layers = len(self._hidden_channels)
        self._paddings = [snt.SAME] * self._num_layers

        self._prob_drop = prob_drop
        self._drop_connect = drop_connect
        self._activation = activation
        self._extra_params = extra_params
        self._aleatoric_layer_tuple = aleatoric_layer
        # instantiate all the convolutional layers

        def custom_getter_dropconnect(getter, name, *args, **kwargs):
            shape = kwargs["shape"]
            theta = getter(name, *args, **kwargs)

            theta_masked = tf.layers.dropout(theta, self._prob_drop, training=True)*(1-self._prob_drop)

            return theta_masked

        # custom_getter = custom_getter_dropconnect if self._drop_connect else None   ##"b" : custom_getter_dropconnect
        self._custom_getter = {"w" : custom_getter_dropconnect} if self._drop_connect else None

        self._bn_renormalization = True
        self._bn_momentum = bn_momentum

    def maybe_dropout(self, net):
        if self._drop_connect == False:
            net = tf.layers.dropout(net, self._prob_drop, training=True)
        return net

    def _build(self, inputs, is_training=True, test_local_stats=True):
        """
        Args:
            inputs (type): node of input.
            is_training (type): tells to batchnorm if to generate the update ops.
            test_local_stats (type): used to test local stats in batch norm.

        Returns:
            logits

        """

        net = inputs

        # self._custom_getter_layer = [self._custom_getter]*(self._num_layers)

        self.layers_one = [snt.Conv2D(name="conv_2d_1_{}".format(i),
                             output_channels=self._hidden_channels[i],
                             kernel_shape=self._kernel_shapes[i],
                             #stride=self._strides[i],
                             padding=self._paddings[i],
                             use_bias=True,
                             custom_getter= self._custom_getter,
                             **self._extra_params
                             ) for i in range(self._num_layers)]


        # connect them to the graph, adding batch norm and non-linearity
        for i, layer_one in enumerate(self.layers_one):

            # LAYER1
            net = layer_one(net)
            net = self.maybe_dropout(net)
            net = self._activation(net)
            if i==0 or i==1 or i==4:
                net = tf.layers.max_pooling2d(net, pool_size=2, strides=self._strides[i], padding="SAME")


        # net=snt.Conv2D(name="conv_2d/l_1",
        #                  output_channels=256,
        #                  kernel_shape=[2,2],
        #                  stride=1,
        #                  padding='VALID',
        #                  use_bias=True,
        #                  custom_getter= self._custom_getter,
        #                  **self._extra_params)(net)
        #
        # net = self.maybe_dropout(net)
        # net = self._activation(net)
        # net= snt.Conv2D(name="conv_2d/l_2",
        #                      output_channels=256,
        #                      kernel_shape=[1,1],
        #                      stride=1,
        #                      padding='SAME',
        #                     use_bias=True,
        #                     custom_getter= self._custom_getter,
        #                     **self._extra_params)(net)
        #
        # net = self.maybe_dropout(net)
        # net = self._activation(net)
        net= snt.Conv2D(name="conv_2d/plain_layer_34",
                              output_channels=32,
                              kernel_shape=[1,1],
                              stride=1,
                              padding='SAME',
                             use_bias=True,
                             **self._extra_params)(net)
        net = snt.BatchFlatten(name="flatten")(net)
        name, kwargs = self._aleatoric_layer_tuple
        aleatoric_kwargs = {
            **self._extra_params,
            "output_size": self._output_size,
            "custom_getter": None #self._custom_getter
        }

        aleatoric_module = network_layer_builder(name, aleatoric_kwargs)

        distr = aleatoric_module(net)

        return distr



    # def _vggconv_block(self, output_channels, kernel, stride, padding, i, **layer_kwargs):
    #     """Network block for VGG."""
    #
    #     conv1 = snt.Conv2D(name="conv_2d_1_{}".format(i),
    #                output_channels = output_channels,
    #                kernel_shape=kernel,
    #                padding=padding,
    #                use_bias=True,
    #                **layer_kwargs
    #                )
    #
    #     net = conv1(net)
    #     bn = snt.BatchNorm(name="batch_norm_1_{}".format(i))
    #     net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
    #     net = self._activation(net)
    #
    #     net = self.maybe_dropout(net)
    #     # if  self._drop_connect== False:
    #     #     net= tf.layers.dropout(net,self._prob_drop, training=True)
    #
    #     net = layer_two(net)
    #     bn = snt.BatchNorm(name="batch_norm_2_{}".format(i))
    #     net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
    #     net = self._activation(net)
    #     net = tf.layers.max_pooling2d(net, 2, self._strides[i], padding="VALID")
    #
    #     net = self.maybe_dropout(net)
    #
    #     layers_list = [
    #         tfp.layers.Convolution2DFlipout(
    #             filters,
    #             kernel,
    #             padding='same',
    #             **layer_kwargs),
    #
    #         tf.keras.layers.BatchNormalization(),
    #
    #         tf.keras.layers.LeakyReLU(),
    #
    #         tfp.layers.Convolution2DFlipout(
    #             filters,
    #             kernel,
    #             padding='same',
    #             **layer_kwargs),
    #
    #         tf.keras.layers.BatchNormalization(),
    #
    #         tf.keras.layers.LeakyReLU(),
    #
    #         tf.keras.layers.MaxPooling2D(
    #             pool_size=(2, 2), strides=stride)]
    #     return layers_list


class Network_WL(snt.AbstractModule):

    def __init__(self, output_size,
                 filters=[16, 16, 32, 32, 32],  # [32, 64, 64, 128, 128],
                 kernels=[3, 3, 3, 3, 3],
                 strides=[2, 2, 2, 2, 2],
                 name="vggB",
                 activation=tf.nn.relu,
                 drop_connect=False,
                 prob_drop=0.1,
                 bn_momentum=0.99,
                 bn_renormalization=True,
                 aleatoric_layer = ("MultivariateNormalDiag", {}),
                 **extra_params):
        """

        Args:
            num_outputs (type): number of outputs of the module.
            name (type): module name.
            activation (type): activation used for the internal layers.
            **extra_params (type): alls the additional keyword arguments will be passed to the snt.Conv2D layers. (initializers, regularizers)

        """
        super().__init__(name=name)
        self._output_size = output_size
        self._hidden_channels = filters
        self._strides = strides
        self._kernel_shapes = kernels  # All kernels are 3x3.
        self._num_layers = len(self._hidden_channels)
        self._paddings = [snt.SAME] * self._num_layers

        self._prob_drop = prob_drop
        self._drop_connect = drop_connect
        self._activation = activation
        self._extra_params = extra_params
        self._aleatoric_layer_tuple = aleatoric_layer
        self._bn_renormalization = bn_renormalization
        # instantiate all the convolutional layers

        def custom_getter_dropconnect(getter, name, *args, **kwargs):
            shape = kwargs["shape"]
            theta = getter(name, *args, **kwargs)

            theta_masked = tf.layers.dropout(theta, self._prob_drop, training=True)

            return theta_masked

        # custom_getter = custom_getter_dropconnect if self._drop_connect else None
        self._custom_getter = {"w" : custom_getter_dropconnect} if self._drop_connect else None


        self._bn_momentum = bn_momentum

        self._renorm_clipping = None
        if self._bn_renormalization:
            self._renorm_clipping=get_renorm_clipping()

    def maybe_dropout(self, net):
        if self._drop_connect == False:
            net = tf.layers.dropout(net, self._prob_drop, training=True)
        return net

    def _build(self, inputs, is_training=True, test_local_stats=True):
        """
        Args:
            inputs (type): node of input.
            is_training (type): tells to batchnorm if to generate the update ops.
            test_local_stats (type): used to test local stats in batch norm.

        Returns:
            logits

        """

        net = inputs
        net= snt.Conv2D(name="conv_2d1",
                              output_channels=self._hidden_channels[0],
                              kernel_shape=self._kernel_shapes[0],
                              padding='SAME',
                             use_bias=True,
                             **self._extra_params)(net)
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=self._bn_momentum,
                                            renorm=self._bn_renormalization,
                                            renorm_momentum=self._bn_momentum,
                                            renorm_clipping=self._renorm_clipping,
                                            name="batch_norm_1")
        net = self._activation(net)
        net= snt.Conv2D(name="conv_2d2",
                              output_channels=self._hidden_channels[0],
                              kernel_shape=self._kernel_shapes[0],
                              padding='SAME',
                             use_bias=True,
                             **self._extra_params)(net)
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=self._bn_momentum,
                                            renorm=self._bn_renormalization,
                                            renorm_momentum=self._bn_momentum,
                                            renorm_clipping=self._renorm_clipping,
                                            name="batch_norm_2")
        net = self._activation(net)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=self._strides[0], padding="VALID")
        net= snt.Conv2D(name="conv_3",
                              output_channels=self._hidden_channels[0]*2,
                              kernel_shape=self._kernel_shapes[0],
                              padding='SAME',
                             use_bias=True,
                             **self._extra_params)(net)
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=self._bn_momentum,
                                            renorm=self._bn_renormalization,
                                            renorm_momentum=self._bn_momentum,
                                            renorm_clipping=self._renorm_clipping,
                                            name="batch_norm_2")
        net = self._activation(net)














        # self._custom_getter_layer = [self._custom_getter]*(self._num_layers)

        self.layers_one = [snt.Conv2D(name="conv_2d_1_{}".format(i),
                             output_channels=self._hidden_channels[i],
                             kernel_shape=self._kernel_shapes[i],
                             #stride=self._strides[i],
                             padding=self._paddings[i],
                             use_bias=True,
                             custom_getter= self._custom_getter,
                             **self._extra_params
                             ) for i in range(self._num_layers)]

        self.layers_two = [snt.Conv2D(name="conv_2d_2_{}".format(i),
                             output_channels=self._hidden_channels[i],
                             kernel_shape=self._kernel_shapes[i],
                            # stride=self._strides[i],
                             padding=self._paddings[i],
                             use_bias=True,
                              custom_getter= self._custom_getter,
                             **self._extra_params
                             ) for i in range(self._num_layers)]

        # connect them to the graph, adding batch norm and non-linearity
        for i, (layer_one, layer_two) in enumerate(zip(self.layers_one, self.layers_two)):

            # LAYER1
            net = layer_one(net)
            # bn = snt.BatchNorm(name="batch_norm_1_{}".format(i))
            # net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
            net = self.maybe_dropout(net)
            net = tf.layers.batch_normalization(net, training=is_training,
                                                momentum=self._bn_momentum,
                                                renorm=self._bn_renormalization,
                                                renorm_momentum=self._bn_momentum,
                                                renorm_clipping=self._renorm_clipping,
                                                name="batch_norm_1_{}".format(i))


            net = self._activation(net)

            # LAYER1

            # LAYER2
            net = layer_two(net)
            # bn = snt.BatchNorm(name="batch_norm_2_{}".format(i))
            # net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
            net = self.maybe_dropout(net)
            net = tf.layers.batch_normalization(net, training=is_training,
                                                momentum=self._bn_momentum,
                                                renorm=self._bn_renormalization,
                                                renorm_momentum=self._bn_momentum,
                                                renorm_clipping=self._renorm_clipping,
                                                name="batch_norm_2_{}".format(i))


            net = self._activation(net)
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=self._strides[i], padding="VALID")
            # LAYER2

        net = snt.BatchFlatten(name="flatten")(net)

        # net = self.maybe_dropout(net)

        # if  self._drop_connect == False:
        #     net= tf.layers.dropout(net,self._prob_drop, training=True)

        name, kwargs = self._aleatoric_layer_tuple
        aleatoric_kwargs = {
            **self._extra_params,
            "output_size": self._output_size,

            "custom_getter": None # self._custom_getter

            # no custom getter on the last layer
            # "custom_getter": self._custom_getter

        }

        self._distr_model = network_layer_builder(name, aleatoric_kwargs)

        distr = self._distr_model(net)

        return distr



class Vgg16(snt.AbstractModule):

  def __init__(self, output_size, name="vgg16", activation=tf.nn.relu, **extra_params):
    """

    Args:
        num_outputs (type): number of outputs of the module.
        name (type): module name.
        activation (type): activation used for the internal layers.
        **extra_params (type): alls the additional keyword arguments will be passed to the snt.Conv2D layers.

    """
    super(Vgg16, self).__init__(name=name)
    self._output_size = output_size
    self._hidden_channels = [
        64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512
        ]
    self._num_layers = len(self._hidden_channels)

    self._kernel_shapes = [[3, 3]] * self._num_layers  # All kernels are 3x3.
    self._strides = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1]
    self._paddings = [snt.SAME] * self._num_layers
    self._activation = activation
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

    net = inputs
    # instantiate all the convolutional layers
    self.layers = [snt.Conv2D(name="conv_2d_{}".format(i),
                         output_channels=self._hidden_channels[i],
                         kernel_shape=self._kernel_shapes[i],
                         stride=self._strides[i],
                         padding=self._paddings[i],
                         use_bias=True,
                         **self._extra_params
                         ) for i in range(self._num_layers)]
    # connect them to the graph, adding batch norm and non-linearity
    for i, layer in enumerate(self.layers):
      net = layer(net)
      bn = snt.BatchNorm(name="batch_norm_{}".format(i))
      net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
      net = self._activation(net)

    net = snt.BatchFlatten(name="flatten")(net)

    logits = snt.Linear(self._output_size)(net)

    return logits


class Vgg19(Vgg16):

  def __init__(self, num_outputs, name="vgg19", activation=tf.nn.relu, **extra_params):
    """

    Args:
        num_outputs (type): number of outputs of the module.
        name (type): module name.
        activation (type): activation used for the internal layers.
        **extra_params (type): alls the additional keyword arguments will be passed to the snt.Conv2D layers.
    """
    super(Vgg19, self).__init__(name=name)
    self._output_size = num_outputs
    self._kernel_shapes = [[3, 3]] * self._num_layers  # All kernels are 3x3.
    self._hidden_channels = [
        64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512
        ]
    self._num_layers = len(self._hidden_channels)
    self._strides = [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
    self._paddings = [snt.SAME] * self._num_layers
    self._activation = activation
    self._extra_params = extra_params



class ResUnit(snt.AbstractModule):
  def __init__(self, depth, name="resUnit", kernel_shape=[3,3], stride=1, activation=tf.nn.relu, **extra_params):
    """
    Args:
        depth (int): the depth of the resUnit.
        name (str): module name.
        kernel_shape (int or [int,int]): the kernel size
        stride (int): the stride
        activation (tf function): activation used for the internal layers.
        **extra_params: all the additional keyword arguments will be passed to snt.Conv2D layers.
    """
    super(ResUnit, self).__init__(name=name)
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
    self.layers = [snt.Conv2D(name="conv_2d_{}".format(i),
                         output_channels=self._depth,
                         kernel_shape=self._kernel_shapes[i],
                         stride=self._strides[i],
                         padding=self._padding,
                         use_bias=True,
                         **self._extra_params
                         ) for i in range(self._num_layers)]
    # connect them to the graph, adding batch norm and non-linearity
    for i, layer in enumerate(self.layers):
      bn = snt.BatchNorm(name="batch_norm_{}".format(i))
      net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
      net = self._activation(net)
      net = layer(net)

    inputstoadd = inputs
    if self._downsample_input:
        inputstoadd = snt.Conv2D(name="conv_2d_fix",
                             output_channels=self._depth,
                             kernel_shape=[1,1],
                             stride=self._strides[0],
                             padding=self._padding,
                             use_bias=True,
                             **self._extra_params
                             )(inputs)

    logits = net + inputstoadd
    return logits


class Res18(snt.AbstractModule):

  def __init__(self, num_outputs, name="res18", activation=tf.nn.relu, **extra_params):
    """
    Args:
        num_outputs (int): the number of outputs of the module.
        name (str): module name.
        activation (tf function): activation used for the internal layers.
        **extra_params: all the additional keyword arguments will be passed to all the snt.Conv2D and to the ResUnit layers.

    """
    super(Res18, self).__init__(name=name)
    self._output_size = num_outputs

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
    self.layers = [snt.Conv2D(name="conv_2d",
                         output_channels=self._conv_channels,
                         kernel_shape=self._conv_kernel_shape,
                         stride=self._conv_stride,
                         padding=self._padding,
                         use_bias=True,
                         **self._extra_params)]
#(self, depth, name="resUnit", kernel_shape=[3,3], stride=1, activation=tf.nn.relu, **extra_params)
    for i in range(self._num_resunits):
        self.layers.append(ResUnit(
                             depth= self._resunit_channels[i],
                             name="resunit{}".format(i),
                             kernel_shape=self._resunit_kernel_shapes[i],
                             stride=self._resunit_strides[i],
                             activation=self._activation,
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



name_short = {
    'Vgg_Bernoulli':          'VGGB',
    'Alex_Bernoulli':          'ALEXDrop',
    'MultivariateNormalDiag': 'D',
    'MultivariateNormalTriL': 'TriL',
}


def get_network_module_id(method_tuple):
    """Creates the id for a network module.

    Args:
        method_tuple (tuple): A tuple composed of : (name of the keras model builder function, kwargs to pass to the function).

    Returns:
        string: the idname of the keras network that we want to concatenate in the output filenames.

    """

    # listWithPoints = lambda x: ".".join(re.sub('[( )\[\]]', '', str(x)).replace(' ', '').split(","))

    method_name = method_tuple[0]
    method_kwargs = method_tuple[1]

    methodid = name_short[method_name]

    if method_name == 'Vgg_Bernoulli':
        methodid += "_dc" + str(int(bool(method_kwargs["drop_connect"])))
        methodid += "_dr" + str(method_kwargs["prob_drop"])
        methodid += "_bnm" + str(method_kwargs["bn_momentum"])
        methodid += "_bnr" + str(int(bool(method_kwargs["bn_renormalization"])))
        methodid += "_" + get_network_module_id(method_kwargs["aleatoric_layer"])
    elif method_name == 'Alex_Bernoulli':
        methodid += "_dc" + str(int(bool(method_kwargs["drop_connect"])))
        methodid += "_dr" + str(method_kwargs["prob_drop"])
        methodid += "_" + get_network_module_id(method_kwargs["aleatoric_layer"])

    elif method_name == 'MultivariateNormalDiag' or method_name == 'MultivariateNormalTriL':
        methodid += "_mc" + str(method_kwargs["minimal_covariance"])

    else:
        print('----------------------')
        print('ERROR ', method_name)
        raise ValueError("id rule for keras network `%s` has to be implemented." % method_name)

    return methodid




# def vgg16(inputs, **default_params):
#     """
#     Create all the nodes of the network vgg16
#
#     Refernces:
#         [1]   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
#
#     Args:
#         inputs: input node
#         default_params: key values to be used by default in the layers
#
#     Returns:
#         logits of the network
#
#     """
#
#     c=0
#     for i in range(4):
#         layer = snt.Conv2D(name="conv2d_{}".format(c)
#         output_channels=64,
#         kernel_shape=
#         )
#         c+=1
#         net = tf.layers.conv2d(inputs, 64, [3, 3], name='conv%d'%c)
#
#     net = slim.max_pool2d(net, [2, 2], scope='pool1')
#
#     net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#     net = slim.max_pool2d(net, [2, 2], scope='pool2')
#
#     net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#     net = slim.max_pool2d(net, [2, 2], scope='pool3')
#     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#     net = slim.max_pool2d(net, [2, 2], scope='pool4')
#     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#     net = slim.max_pool2d(net, [2, 2], scope='pool5')
#     net = slim.fully_connected(net, 1000, activation_fn=tf.nn.relu,
#                                scope='fc0')
#     net = slim.flatten(net, scope='flatten1')
#     net = slim.fully_connected(net, 1, activation_fn=None, scope='fc1')
#     return net


# def vgg19(inputs, activation_fn=tf.nn.relu,
#           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#           weights_regularizer=slim.l2_regularizer(0.0005)):
#     """
#        Create all the nodes of the network vgg19
#
#        Refernces:
#            [1]   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
#
#        Args:
#            inputs: input node
#
#        Returns:
#            logits of the network
#
#     """
#
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         activation_fn=activation_fn,
#                         weights_initializer=weights_initializer,
#                         weights_regularizer=weights_regularizer):
#         net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#         net = slim.max_pool2d(net, [2, 2], scope='pool1')
#         net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#         net = slim.max_pool2d(net, [2, 2], scope='pool2')
#         net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
#         net = slim.max_pool2d(net, [2, 2], scope='pool3')
#         net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
#         net = slim.max_pool2d(net, [2, 2], scope='pool4')
#         net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
#         net = slim.max_pool2d(net, [2, 2], scope='pool5')
#         net = slim.fully_connected(net, 4096, activation_fn=tf.nn.relu, scope='fc0')
#         net = slim.fully_connected(net, 4096, activation_fn=tf.nn.relu, scope='fc1')
#         net = slim.flatten(net,  scope='flatten1')
#         net = slim.fully_connected(net, 1, activation_fn=None, scope='fc2')
#     return net

# def plain_layer_34(inputs, activation_fn=tf.nn.relu,
#           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#           weights_regularizer=slim.l2_regularizer(0.0005)):
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         activation_fn=activation_fn,
#                         weights_initializer=weights_initializer,
#                         weights_regularizer=weights_regularizer):
#         layer1 = slim.conv2d(inputs, 64, [7, 7], stride=[2, 2],
#                              normalizer_fn=slim.batch_norm,
#                              scope='conv_' + str(0))
#         layer1 = slim.max_pool2d(layer1, [2, 2], scope='pool_' + str(0))
#         layer1 = slim.repeat(layer1, 6, slim.conv2d, 64, [3, 3],
#                              scope='conv_s_3')
#         layer1 = slim.conv2d(layer1, 128, [3, 3], stride=[2, 2],
#                              normalizer_fn=slim.batch_norm, scope='conv_4')
#         layer1 = slim.repeat(layer1, 7, slim.conv2d, 128, [3, 3],
#                              scope='conv_s_5')
#         layer1 = slim.conv2d(layer1, 256, [3, 3], stride=[2, 2],
#                              normalizer_fn=slim.batch_norm, scope='conv_6')
#         layer1 = slim.repeat(layer1, 11, slim.conv2d, 256, [3, 3],
#                              scope='conv_s_7')
#         layer1 = slim.conv2d(layer1, 512, [3, 3], stride=[2, 2],
#                              normalizer_fn=slim.batch_norm, scope='conv_8')
#         layer1 = slim.repeat(layer1, 5, slim.conv2d, 512, [3, 3],
#                              scope='conv_s_9')
#         layer1 = slim.avg_pool2d(layer1, [2, 2], scope='ave_pool_' + str(0))
#         output = slim.layers.flatten(layer1, scope='flatten' + str(0))
#         output = slim.fully_connected(output, 1, activation_fn=None,
#                                       scope='fc' + str(1))
#     return output

# def res_18(inputs, activation_fn=tf.nn.relu,
#           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#           weights_regularizer=slim.l2_regularizer(0.0005)):
#     total_layers = 36  # Specify how deep we want our network
#     units_between_stride = int(total_layers / 4)
#
#     def resUnit(input_layer, depth, size_filter, stride_stp, i):
#         with tf.variable_scope("res_unit" + str(i)):
#             part1 = slim.batch_norm(input_layer, activation_fn=None)
#             part2 = tf.nn.relu(part1)
#             part3 = slim.conv2d(part2, depth, [size_filter, size_filter],
#                                 stride=[stride_stp, stride_stp],
#                                 activation_fn=None)
#             part4 = slim.batch_norm(part3, activation_fn=None)
#             part5 = tf.nn.relu(part4)
#             part6 = slim.conv2d(part5, depth, [size_filter, size_filter],
#                                 activation_fn=None)
#             dims_dep = input_layer.shape[3]
#             dims_ima = input_layer.shape[2]
#
#             if (dims_ima == part6.shape[2] and dims_dep == depth):
#                 out = input_layer + part6
#
#             else:
#                 input_layer1 = slim.conv2d(input_layer, depth, [1, 1],
#                                            stride=[stride_stp, stride_stp],
#                                            activation_fn=None)
#                 out = input_layer1 + part6
#
#         return out
#
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         activation_fn=activation_fn,
#                         weights_initializer=weights_initializer,
#                         weights_regularizer=weights_regularizer):
#
#         layer1 = slim.conv2d(inputs, 64, [7, 7], stride=[2, 2],
#                              normalizer_fn=slim.batch_norm,
#                              scope='conv_s_' + str(0))
#         layer1 = slim.max_pool2d(layer1, [2, 2], scope='pool_' + str(0))
#
#         layer1 = resUnit(layer1, 64, 3, 1, 0)
#         layer1 = resUnit(layer1, 64, 3, 1, 1)
#
#         layer1 = resUnit(layer1, 128, 3, 2, 2)
#         layer1 = resUnit(layer1, 128, 3, 1, 3)
#
#         layer1 = resUnit(layer1, 256, 3, 2, 4)
#         layer1 = resUnit(layer1, 256, 3, 1, 5)
#
#         layer1 = resUnit(layer1, 512, 3, 2, 6)
#         layer1 = resUnit(layer1, 512, 3, 1, 7)
#
#         layer1 = slim.avg_pool2d(layer1, [2, 2], scope='ave_pool_' + str(0))
#         output = slim.layers.flatten(layer1, scope='flatten' + str(0))
#         output = slim.fully_connected(output, 1, activation_fn=None,
#                                       scope='fc' + str(1))
#
#     return output
#
# def res_34(inputs, activation_fn=tf.nn.relu,
#           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#           weights_regularizer=slim.l2_regularizer(0.0005)):
#     total_layers = 36  # Specify how deep we want our network
#     units_between_stride = int(total_layers / 4)
#
#     def resUnit(input_layer, depth, size_filter, stride_stp, i):
#         with tf.variable_scope("res_unit" + str(i)):
#             part1 = slim.batch_norm(input_layer, activation_fn=None)
#             part2 = tf.nn.relu(part1)
#             part3 = slim.conv2d(part2, depth, [size_filter, size_filter],
#                                 stride=[stride_stp, stride_stp],
#                                 activation_fn=None)
#             part4 = slim.batch_norm(part3, activation_fn=None)
#             part5 = tf.nn.relu(part4)
#             part6 = slim.conv2d(part5, depth, [size_filter, size_filter],
#                                 activation_fn=None)
#             dims_dep = input_layer.shape[3]
#             dims_ima = input_layer.shape[2]
#
#             if (dims_ima == part6.shape[2] and dims_dep == depth):
#                 out = input_layer + part6
#
#             else:
#                 input_layer1 = slim.conv2d(input_layer, depth, [1, 1],
#                                            stride=[stride_stp, stride_stp],
#                                            activation_fn=None)
#                 out = input_layer1 + part6
#
#         return out
#
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         activation_fn=activation_fn,
#                         weights_initializer=weights_initializer,
#                         weights_regularizer=weights_regularizer):
#         layer1 = slim.conv2d(inputs, 64, [7, 7], stride=[2, 2],
#                              normalizer_fn=slim.batch_norm,
#                              scope='conv_s_' + str(0))
#         layer1 = slim.max_pool2d(layer1, [2, 2], scope='pool_' + str(0))
#
#         layer1 = resUnit(layer1, 64, 3, 1, 0)
#         layer1 = resUnit(layer1, 64, 3, 1, 1)
#         layer1 = resUnit(layer1, 64, 3, 1, 2)
#
#         layer1 = resUnit(layer1, 128, 3, 2, 3)
#         layer1 = resUnit(layer1, 128, 3, 1, 4)
#         layer1 = resUnit(layer1, 128, 3, 1, 5)
#         layer1 = resUnit(layer1, 128, 3, 1, 6)
#
#         layer1 = resUnit(layer1, 256, 3, 2, 7)
#         layer1 = resUnit(layer1, 256, 3, 1, 8)
#         layer1 = resUnit(layer1, 256, 3, 1, 9)
#         layer1 = resUnit(layer1, 256, 3, 1, 10)
#         layer1 = resUnit(layer1, 256, 3, 1, 11)
#         layer1 = resUnit(layer1, 256, 3, 1, 12)
#
#         layer1 = resUnit(layer1, 512, 3, 2, 13)
#         layer1 = resUnit(layer1, 512, 3, 1, 14)
#         layer1 = resUnit(layer1, 512, 3, 1, 15)
#
#         layer1 = slim.avg_pool2d(layer1, [2, 2], scope='ave_pool_' + str(0))
#         output = slim.layers.flatten(layer1, scope='flatten' + str(0))
#         output = slim.fully_connected(output, 1, activation_fn=None,
#                                       scope='fc' + str(1))
#
#     return output
#
#
# def res_50(inputs, activation_fn=tf.nn.relu,
#           weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#           weights_regularizer=slim.l2_regularizer(0.0005)):
#     total_layers = 36  # Specify how deep we want our network
#     units_between_stride = int(total_layers / 4)
#
#     def resUnit(input_layer, depth_ini, depth_end, size_filter, stride_stp, i):
#         with tf.variable_scope("res_unit" + str(i)):
#             part1 = slim.batch_norm(input_layer, activation_fn=None)
#             part2 = tf.nn.relu(part1)
#             part3 = slim.conv2d(part2, depth_ini, [1, 1],
#                                 stride=[stride_stp, stride_stp],
#                                 activation_fn=None)
#             part4 = slim.batch_norm(part3, activation_fn=None)
#             part5 = tf.nn.relu(part4)
#             part6 = slim.conv2d(part5, depth_ini, [size_filter, size_filter],
#                                 activation_fn=None)
#             part7 = slim.batch_norm(part6, activation_fn=None)
#             part8 = tf.nn.relu(part7)
#             part9 = slim.conv2d(part8, depth_end, [1, 1], activation_fn=None)
#
#             dims_dep = input_layer.shape[3]
#             dims_ima = input_layer.shape[2]
#
#             if (dims_ima == part6.shape[2] and dims_dep == depth):
#                 out = input_layer + part9
#
#             else:
#                 input_layer1 = slim.conv2d(input_layer, depth_end, [1, 1],
#                                            stride=[stride_stp, stride_stp],
#                                            activation_fn=None)
#                 out = input_layer1 + part9
#
#         return out
#
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         activation_fn=activation_fn,
#                         weights_initializer=weights_initializer,
#                         weights_regularizer=weights_regularizer):
#
#         layer1 = slim.conv2d(inputs, 64, [7, 7], stride=[2, 2],
#                              normalizer_fn=slim.batch_norm,
#                              scope='conv_s_' + str(0))
#         layer1 = slim.max_pool2d(layer1, [2, 2], scope='pool_' + str(0))
#
#         layer1 = resUnit(layer1, 64, 256, 3, 2, 0)
#         layer1 = resUnit(layer1, 64, 256, 3, 1, 1)
#         layer1 = resUnit(layer1, 64, 256, 3, 1, 2)
#
#         layer1 = resUnit(layer1, 128, 512, 3, 2, 3)
#         layer1 = resUnit(layer1, 128, 512, 3, 1, 4)
#         layer1 = resUnit(layer1, 128, 512, 3, 1, 5)
#         layer1 = resUnit(layer1, 128, 512, 3, 1, 6)
#
#         layer1 = resUnit(layer1, 256, 1024, 3, 2, 7)
#         layer1 = resUnit(layer1, 256, 1024, 3, 1, 8)
#         layer1 = resUnit(layer1, 256, 1024, 3, 1, 9)
#         layer1 = resUnit(layer1, 256, 1024, 3, 1, 10)
#         layer1 = resUnit(layer1, 256, 1024, 3, 1, 11)
#         layer1 = resUnit(layer1, 256, 1024, 3, 1, 12)
#
#         layer1 = resUnit(layer1, 512, 2048, 3, 2, 13)
#         layer1 = resUnit(layer1, 512, 2048, 3, 1, 14)
#         layer1 = resUnit(layer1, 512, 2048, 3, 1, 15)
#
#         layer1 = slim.avg_pool2d(layer1, [2, 2], scope='ave_pool_' + str(0))
#         output = slim.layers.flatten(layer1, scope='flatten' + str(0))
#         output = slim.fully_connected(output, 1, activation_fn=None,
#                                       scope='fc' + str(1))
#
#     return output
