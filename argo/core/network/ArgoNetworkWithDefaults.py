import tensorflow as tf

from .ArgoAbstractNetwork import ArgoAbstractNetwork
from .initializers.TFInitializers import TFInitializers
from ..utils.argo_utils import method_name_short, get_method_id, eval_method_from_tuple


class ArgoNetworkWithDefaults(ArgoAbstractNetwork):
    """
    Abstract class for managing a network, providing defaults,
    like: initializations, regularizations and activation...
    """

    default_params = {
        **ArgoAbstractNetwork.default_params,
        "weights_init": ("contrib.layers.xavier_initializer", {}),
        "bias_init":    ("constant_initializer", {
            "value": 0.1}),
        "weights_reg":  None,
        "bias_reg":     None,
        "activation":   "relu"  # you can also try "relu", "sigmoid"
    }

    def create_id(self):

        _id = '-a' + method_name_short[self._opts["activation"]] + \
              '-wi' + TFInitializers.create_id(self._opts["weights_init"]) + \
              '-bi' + TFInitializers.create_id(self._opts["bias_init"])

        if self._opts["weights_reg"]:
            _id += '-wr' + get_method_id(self._opts["weights_reg"])

        if self._opts["bias_reg"]:
            _id += '-br' + get_method_id(self._opts["bias_reg"])

        super_id = super().create_id()

        _id += super_id

        return _id

    # it is in an extra class because Not every network
    # should be forced to specify these parameters for layers
    # e.g. predefined architectures and others
    # TODO is this the place for the seed? It should be where the session is created, not here...
    def __init__(self, opts, name, seed=None):
        super().__init__(opts, name, seed)

        # GET DEFAULTS FOR LAYERS CREATION
        # initializers
        self._default_weights_init = TFInitializers.instantiate_initializer(self._opts["weights_init"])
        self._default_bias_init = TFInitializers.instantiate_initializer(self._opts["bias_init"])

        # activation function
        self._default_activation = getattr(tf.nn, self._opts["activation"])

        # regularizers
        self._default_weights_reg = eval_method_from_tuple(tf, self._opts["weights_reg"])
        self._default_bias_reg = eval_method_from_tuple(tf, self._opts["bias_reg"])

        self._network_defaults = {
            "activation":   self._default_activation,
            "weights_init": self._default_weights_init,
            "bias_init":    self._default_bias_init,
            "weights_reg":  self._default_weights_reg,
            "bias_reg":     self._default_bias_reg
        }