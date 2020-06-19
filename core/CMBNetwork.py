from argo.core.network.ArgoNetworkWithDefaults import ArgoNetworkWithDefaults
from argo.core.utils.utils_modules import get_network_module_id

import importlib


class CMBNetwork(ArgoNetworkWithDefaults):
    """
    Network class for managing the network used for CMB
    """

    default_params = {
        **ArgoNetworkWithDefaults.default_params,
        "network_architecture" : [("Vgg_Bernoulli", {"prob_drop": 0.1,
                                                    "drop_connect" : False,
                                                    "aleatoric_layer" : ("MultivariateNormalDiag", {})
                                                   }
                                    )
                                   ]
        }

    def create_id(self):
        
        _id = "-" + get_network_module_id(self._opts["network_architecture"])

        super_id = super().create_id()

        _id += super_id
        return _id

    def __init__(self, opts, name): # , remove_node_from_logits = False
        super().__init__(opts, name)
        self._network_architecture = self._opts["network_architecture"]
        self._module_name, self._module_kwargs = self._network_architecture

        self._output_size = self._opts["output_shape"][0]
        self._input_shape = self._opts["input_shape"]

        self._activation = self._network_defaults["activation"]
        self._initializers = {'w' : self._network_defaults["weights_init"],
                        'b' : self._network_defaults["bias_init"]}
        self._regularizers = {'w' : self._network_defaults["weights_reg"],
                        'b' : self._network_defaults["bias_reg"]}

        utils_module = importlib.import_module(".argo.core.utils.utils_modules", '.'.join(__name__.split('.')[:-1]))
        self._module_class = getattr(utils_module, self._module_name)

        # self._network_defaults = {
        #         "activation" : self._default_activation,
        #         "weights_init" : self._default_weights_init,
        #         "bias_init" : self._default_bias_init,
        #         "weights_reg" : self._default_weights_reg,
        #         "bias_reg" : self._default_bias_reg
        # }
        #


    def _build(self, x, is_training=None, **extra_kwargs):

        self.module = self._module_class(**self._module_kwargs,
                                        activation=self._activation,
                                        initializers=self._initializers,
                                        regularizers=self._regularizers,
                                        output_size=self._output_size
                                        )


        return self.module(x, is_training = is_training)
