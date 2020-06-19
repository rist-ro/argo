from .ArgoNetworkWithDefaults import ArgoNetworkWithDefaults
from ..utils.argo_utils import get_method_id


class ArgoStochasticNetworkWithDefaults(ArgoNetworkWithDefaults):
    """
    Abstract class for managing a stochastic network, providing defaults,
    like: covariance_parameterization, contractive_regularizer, ...
    """

    default_params = {
        **ArgoNetworkWithDefaults.default_params,

        "covariance_parameterization": "softplus",  # softplus or exp
    }

    def create_id(self):
        _id = '-cp' + get_method_id((self._opts["covariance_parameterization"], {}))

        super_id = super().create_id()

        _id += super_id

        return _id

    # it is in an extra class because Not every network
    # should be forced to specify these parameters for stochastic layers
    # e.g. predefined architectures and others
    def __init__(self, opts, name, seed):
        super().__init__(opts, name, seed)

        # GET DEFAULTS FOR STOCHASTIC LAYERS
        default_covariance_parametrization = self._opts["covariance_parameterization"]

        self._stochastic_defaults = {
            "covariance_parameterization": default_covariance_parametrization,
        }

    #     # these are default parameters for the layers, "potentially" they could be overwritten
    #     # by the properties in the specific layers of "network_archicture"
    #     self._default_layers_kwargs = {}
    #
    #     self._default_layers_kwargs['common'] = {
    #         "activation" : default_activation,
    #         "kernel_initializer" : default_weights_init,
    #         "bias_initializer" : default_bias_init,
    #         "kernel_regularizer" : default_weights_reg,
    #         "bias_regularizer" : default_bias_reg,
    #     }
    #
    #     self._default_layers_kwargs['flatten'] = {}
    #
    #     self._default_layers_kwargs['dense'] = {
    #         "activation" : default_activation,
    #         "kernel_initializer" : default_weights_init,
    #         "bias_initializer" : default_bias_init,
    #         "kernel_regularizer" : default_weights_reg,
    #         "bias_regularizer" : default_bias_reg,
    #         "activity_regularizer" : None,
    #         "kernel_constraint" : None,
    #         "bias_constraint" : None
    #     }
    #
    #     self._default_layers_kwargs['conv2d'] = {
    #         **self._default_layers_kwargs['dense'],
    #         # filters,
    #         # kernel_size,
    #         "strides" : (1, 1),
    #         "padding" : 'valid'
    #         # "data_format" : 'channels_last',
    #         # "dilation_rate" : (1, 1)
    #     }
    #
    #     self._default_layers_kwargs['max_pooling2d'] = {
    #         "pool_size": (2,2),
    #         "strides": 2
    #     }
    #
    #     self._default_layers_kwargs['batch_normalization'] = {
    #     }
    #
    #
    # def _create_deterministic_layers(self, x, network_architecture, name=None):
    #     """Create the logits nodes of the network.
    #
    #     Sets:
    #         to the created corresponding tf variables
    #
    #     Args:
    #         x (tf.placeholder): input node.
    #         weights (list of np.array): weights of the network.
    #         biases (list of np.array): biases of the network.
    #
    #     Returns:
    #         tf.node: logits node.
    #
    #     """
    #
    #     net = x
    #
    #     # TODO not sure if we need this or not
    #     '''
    #     scope = self._name
    #     if name is not None:
    #         scope += "/" + name
    #
    #     with tf.variable_scope(scope):
    #     '''
    #
    #     for i, layer_tuple in enumerate(network_architecture):
    #         layer_name, layer_kwargs = layer_tuple
    #         layer_fn = getattr(tf.layers, layer_name)
    #         kwargs = {**self._default_layers_kwargs[layer_name],
    #                   **layer_kwargs,
    #                   "name" : "layer"+str(i)+"-"+layer_name
    #                   }
    #         net = layer_fn(net, **kwargs)
    #
    #     # TODO understand when I need this
    #     #self.variables = tf.get_default_graph().get_collection('variables',scope=scope)
    #
    #     return net
    #
    # @classmethod
    # def network_id(cls, opts):
    #     """
    #     generator of part of the id related to the topology of the network
    #     Args:
    #         opts:
    #
    #     Returns:
    #
    #     """
    #
    #
    #     first_net_option = opts["network_architecture"][0]
    #     topology_id = "-"
    #     if isinstance(first_net_option, str):
    #         topology_id += first_net_option
    #     else:
    #         topology_id += '_'.join(get_method_id(layer_tuple)
    #                                 for layer_tuple in opts["network_architecture"]
    #                                     if layer_tuple[0] is not 'flatten')
    #
    #     topology_id += '-wi' + get_method_id(opts["weights_init"]) + \
    #                   '-bi' + get_method_id(opts["bias_init"]) + \
    #                   '-a' + opts["activation"]
    #
    #     return topology_id