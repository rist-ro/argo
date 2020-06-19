from argo.core.utils.argo_utils import get_method_id, load_method_from_tuple

import pdb

class DirectedGraph():
    """
    Class for managing directed graphs
    """

    def __init__(self, opts):
        #GET DEFAULTS FOR LAYERS CREATION
        #initializers
        default_weights_init = load_method_from_tuple(tf, opts["weights_init"])
        default_bias_init = load_method_from_tuple(tf, opts["bias_init"])
        
        #activation function
        default_activation_fct =  getattr(tf.nn, opts["activation"])
        
        #regularizers
        default_weights_reg = load_method_from_tuple(tf, opts["weights_reg"])
        default_bias_reg = load_method_from_tuple(tf, opts["bias_reg"])
        
        # these are default parameters, "potentially" they could be overwritten
        # by possible properties in the specific layers in "network_archicture"
        
        self._default_layers_kwargs = {}

        self._default_layers_kwargs['flatten'] = {}

        self._default_layers_kwargs['dense'] = {
                                "activation" : default_activation_fct,
                                "kernel_initializer" : default_weights_init,
                                "bias_initializer" : default_bias_init,
                                "kernel_regularizer" : default_weights_reg,
                                "bias_regularizer" : default_bias_reg,
                                "activity_regularizer" : None,
                                "kernel_constraint" : None,
                                "bias_constraint" : None
                                }

        self._default_layers_kwargs['conv2d'] = {
                                **self._default_layers_kwargs['dense'],
                                # filters,
                                # kernel_size,
                                "strides" : (1, 1),
                                "padding" : 'valid'
                                # "data_format" : 'channels_last',
                                # "dilation_rate" : (1, 1)
                                }

        self._default_layers_kwargs['max_pooling2d'] = {
            "pool_size": (2,2),
            "strides": 2
        }

        first_net_option = opts["network_architecture"][0]
        if isinstance(first_net_option, str):
            self._network_architecture = opts["network_architecture"][1:]
            self._network_name = first_net_option
        else:
            # parameters which define the network topology
            self._network_architecture = opts["network_architecture"]
            self._network_name = None
            
        
    def create_logits(self, x):
        """Create the logits nodes of the network.

        Sets:
            to the created corresponding tf variables

        Args:
            x (tf.placeholder): input node.
            weights (list of np.array): weights of the network.
            biases (list of np.array): biases of the network.

        Returns:
            tf.node: logits node.

        """

        net = x

        for i, layer_tuple in enumerate(self._network_architecture):
            layer_name, layer_kwargs = layer_tuple
            layer_fn = getattr(tf.layers, layer_name)
            tot_kwargs = {
                    **self._default_layers_kwargs[layer_name],
                    **layer_kwargs,
                    "name" : "layer"+str(i)+"-"+layer_name
                    }
            net = layer_fn(net, **tot_kwargs)

        self.variables = tf.get_default_graph().get_collection('variables')
        return net

    @classmethod
    def algorithm_topology_id(cls, opts):
        """
        generator of part of the id related to the topology of the network
        Args:
            opts:

        Returns:

        """

               
        first_net_option = opts["network_architecture"][0]
        topology_id = "-"
        if isinstance(first_net_option, str):
            topology_id += first_net_option
        else:
            topology_id += '_'.join(get_method_id(layer_tuple)
                                    for layer_tuple in opts["network_architecture"]
                                        if layer_tuple[0] is not 'flatten')

        topology_id += '-wi' + get_method_id(opts["weights_init"]) + \
                      '-bi' + get_method_id(opts["bias_init"]) + \
                      '-a' + opts["activation"]

        return topology_id
    
    @property
    def default_activation_function(self):
        return self.activation_fct

    @property
    def number_layers(self):
        return len(self._network_architecture)
