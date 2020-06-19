import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .AbstractModule import AbstractModule
from ..utils.argo_utils import load_sonnet_module


class GeneralSonnetNetwork(AbstractModule):
    """Intermediate class to build a customizable Network (to be used as a building block from an ArgoNetwork)
    """

    def __init__(self, activation, default_weights_init,
                 default_bias_init, default_weights_reg, default_bias_reg,
                 network_architecture,
                 stochastic_defaults=None,
                 network_str="",
                 is_training=False,
                 # seed=None,
                 name='AbstractSonnetNetwork'):

        """Short summary.
        Args:
            activation (type): Description of parameter `activation`.
            default_weights_init (type): Description of parameter `default_weights_init`.
            default_bias_init (type): Description of parameter `default_bias_init`.
            default_weights_reg (type): Description of parameter `default_weights_reg`.
            default_bias_reg (type): Description of parameter `default_bias_reg`.
            network_architecture (list): a list of tuples (sntModule, kwargs, bool_activate).
                i.e. [(Linear, {"output_size": 100}, 1), (Linear, {"output_size": 10}, 1),
                        ("GaussianDiagonal", {"size" : 20, "minimal_covariance" : 0}, 0)]
            stochastic_defaults (type): Description of parameter `stochastic_defaults`.
            network_str (str): Optional network_str specifying the network we are going to build.
                            It is used to set some specific collections for activity and contractive regularizers.
            name (str): name of the Module.

        """

        super().__init__(name=name)

        self._network_architecture = network_architecture
        self._network_str = network_str

        # self._is_training = is_training
        # self._seed = seed

        covariance_parameterization = None
        concentration_parameterization = None

        if stochastic_defaults:
            try:
                # what to do for multiple sampling layers, now I sample once!
                # self.n_z_samples = stochastic_defaults["samples"]
                if "covariance_parameterization" in stochastic_defaults:
                    covariance_parameterization = stochastic_defaults["covariance_parameterization"]
                elif "concentration_parameterization" in stochastic_defaults:
                    concentration_parameterization = stochastic_defaults["concentration_parameterization"]
            except KeyError as e:
                print("need to pass all the stochastic_defaults, missing keys")
                raise KeyError("need to pass all the stochastic_defaults, missing keys") from e

        # SET DEFAULTS FOR NETWORK CREATION

        # activation function
        self._activation = activation

        # SET DEFAULT PARAMETERS FOR SONNET MODULES
        # these are default parameters for the snt modules, "potentially" they could be overwritten
        # by the properties in the specific modules of "network_archicture"
        self._default_modules_kwargs = {}

        self._default_modules_kwargs['common'] = {
            "initializers": {},
            "regularizers": {}
        }
        # "initializers" : {'w':default_weights_init, 'b':default_bias_init},
        # "regularizers" : {'w':default_weights_reg, 'b':default_bias_reg}

        if default_weights_init:
            self._default_modules_kwargs['common']["initializers"]['w'] = default_weights_init
        if default_bias_init:
            self._default_modules_kwargs['common']["initializers"]['b'] = default_bias_init

        if default_weights_reg:
            self._default_modules_kwargs['common']["regularizers"]['w'] = default_weights_reg
        if default_bias_reg:
            self._default_modules_kwargs['common']["regularizers"]['b'] = default_bias_reg

        self._default_modules_kwargs['BatchFlatten'] = {}

        self._default_modules_kwargs['BatchReshape'] = {}

        self._default_modules_kwargs['Sigmoid'] = {}

        self._default_modules_kwargs['Tanh'] = {}
 
        self._default_modules_kwargs['Linear'] = {
            **self._default_modules_kwargs['common']
        }

        self._default_modules_kwargs['Concatenate'] = {
            "node_name" : "y1h"
         }

        self._default_modules_kwargs['Identity'] = {
         }

                
        self._default_modules_kwargs['Conv2D'] = {
            **self._default_modules_kwargs['Linear'],
            "kernel_shape": (3, 3),
            "stride":       (1, 1),
            "padding":      'SAME'
        }

        self._default_modules_kwargs['Conv2DTranspose'] = {
            **self._default_modules_kwargs['Linear'],
            "kernel_shape": (3, 3),
            "stride":       (1, 1),
            "padding":      'SAME'
        }

        self._default_modules_kwargs['LinearWN'] = {**self._default_modules_kwargs['Linear'],
                                                    "use_weight_norm": True
                                                    }
        # setting default values for the initializers for WN
        self._default_modules_kwargs['LinearWN']["initializers"] = {
            'v': tf.random_normal_initializer(0, 0.05),
            'b': default_bias_init,
            'g': default_bias_init
        }

        self._default_modules_kwargs['Conv2DWN'] = {**self._default_modules_kwargs['Conv2D'],
                                                    "use_weight_norm": True}
        # setting default values for the initializers for WN
        self._default_modules_kwargs['Conv2DWN']["initializers"] = {
            'v': tf.random_normal_initializer(0, 0.05),
            'b': default_bias_init,
            'g': default_bias_init
        }

        self._default_modules_kwargs['custom'] = {**self._default_modules_kwargs['common'],
                                                  "activation": self._activation,
                                                  "is_training": is_training}

        self._default_modules_kwargs['ResUnit'] = {**self._default_modules_kwargs['custom']}
        self._default_modules_kwargs['ResNet18'] = {**self._default_modules_kwargs['custom']}
        self._default_modules_kwargs['VGGBlock'] = {**self._default_modules_kwargs['custom']}
        self._default_modules_kwargs['ConvDec'] = {**self._default_modules_kwargs['custom']}
        self._default_modules_kwargs['ResEnc'] = {**self._default_modules_kwargs['custom']}
        self._default_modules_kwargs['ResDec'] = {**self._default_modules_kwargs['custom']}

        self._default_modules_kwargs['ConvNet2D'] = {
            **self._default_modules_kwargs['Linear'],
            "kernel_shapes":   [(3, 3)],
            "strides":         [2, 2],
            "paddings":        'SAME',
            "activation":      self._activation,
            "activate_final":  False,
            "normalize_final": False
        }

        self._default_modules_kwargs['ConvNet2DTranspose'] = {
            **self._default_modules_kwargs['ConvNet2D']
        }

        self._default_modules_kwargs['MaxPooling2D'] = {
            "pool_size": (2, 2),
            "strides":   2
        }

        self._default_modules_kwargs['RandomUniform'] = {
            "shape":  20,
            "minval": -1,
            "maxval": 1,
        }

        self._default_modules_kwargs['RandomGaussian'] = {
            "shape": 20
        }

        self._default_modules_kwargs['AveragePooling2D'] = {
            "pool_size": (2, 2),
            "strides":   2
        }

        self._default_modules_kwargs['Dropout'] = {
            # "seed" : self._seed,
            # "rate" : 0.5,
            "rate":         0.5,  # tf.layers.dropout
            # "is_training" : self._is_training
            "dropout_flag": is_training

        }

        self._default_modules_kwargs['BatchNorm'] = {
            "is_training": is_training}

        self._default_modules_kwargs['GaussianDiagonal'] = {
            **self._default_modules_kwargs['common'],
            # TODO check the default value
            "minimal_covariance":          0.,
            "covariance_parameterization": covariance_parameterization,
            # This wouls be desiderable, but at the moment it does not work
            # "module_tuple" : ("Linear", {})
        }

        self._default_modules_kwargs['GaussianDiagonalZeroOne'] = {
            **self._default_modules_kwargs['GaussianDiagonal'],
            "module_tuple": ("Linear", {})
        }

        self._default_modules_kwargs['GaussianDiagonalPlusMinusOne'] = {
            **self._default_modules_kwargs['GaussianDiagonal'],
            "module_tuple": ("Linear", {})
        }

        self._default_modules_kwargs['Gaussian'] = {
            **self._default_modules_kwargs['common'],
            # TODO check the default value
            "minimal_covariance":          0.,
            "covariance_parameterization": covariance_parameterization,
            # This wouls be desiderable, but at the moment it does not work
            # "module_tuple" : ("Linear", {})
        }

        self._default_modules_kwargs['vonMisesFisher'] = {
            **self._default_modules_kwargs['common'],
            # TODO check the default value
            # see https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/distributions/von_mises_fisher.py
            "minimal_concentration":          1,
            "concentration_parameterization": concentration_parameterization,
            # This wouls be desiderable, but at the moment it does not work
            # "module_tuple" : ("Linear", {})
        }

        self._default_modules_kwargs['LogisticDiagonalZeroOne'] = {
            **self._default_modules_kwargs['GaussianDiagonalZeroOne'],
        }

        self._default_modules_kwargs['LogisticDiagonalPlusMinusOne'] = {
            **self._default_modules_kwargs['GaussianDiagonalPlusMinusOne'],
        }

        self._default_modules_kwargs['LogitNormalDiagonal'] = {
            **self._default_modules_kwargs['GaussianDiagonalZeroOne'],
            "clip_value": 0.0001
        }

        self._default_modules_kwargs['LogitNormalDiagonalPlusMinusOne'] = {
            **self._default_modules_kwargs['GaussianDiagonalPlusMinusOne'],
            "clip_value": 0.0001
        }

        self._default_modules_kwargs['Bernoulli'] = {
            **self._default_modules_kwargs['common'],
            "clip_value": 0.0001
        }

        self._default_modules_kwargs['BernoulliPlusMinusOne'] = {
            **self._default_modules_kwargs['common'],
            "clip_value": 0.0001
        }

        self._default_modules_kwargs['CIFAR10TutorialNetwork'] = {
        }

        # these are default parameters for the layers, "potentially" they could be overwritten
        # by the properties in the specific layers of "network_archicture"
        self._default_layers_kwargs = {}

        self._default_layers_kwargs['common'] = {
            "kernel_initializer": default_weights_init,
            "bias_initializer":   default_bias_init,
            "kernel_regularizer": default_weights_reg,
            "bias_regularizer":   default_bias_reg,
        }

        self._default_layers_kwargs['flatten'] = {}

        self._default_layers_kwargs['dense'] = {
            **self._default_layers_kwargs['common'],
            "activity_regularizer": None,
            "kernel_constraint":    None,
            "bias_constraint":      None
        }

        self._default_layers_kwargs['conv2d'] = {
            **self._default_layers_kwargs['dense'],
            # filters,
            # kernel_size,
            "strides": (1, 1),
            "padding": 'valid'
            # "data_format" : 'channels_last',
            # "dilation_rate" : (1, 1)
        }

        self._default_layers_kwargs['max_pooling2d'] = {
            "pool_size": (2, 2),
            "strides":   2
        }

        self._default_layers_kwargs['batch_normalization'] = {
        }

    def _build(self, inputs):

        """Constructs the graph

        Args:
          inputs: `tf.Tensor` input to which to attach the network
          network_str: string used to separate collections of the network,
                        i.e. used for activity and contractive regularizers

        Returns:
          if stochastic_architecture==None:
              `tf.Tensor` output with the decision of the discriminator
          else:
              tf.distribution
        """

        print("Parsing " + self.module_name + " network...")

        net = inputs

        self._modules = []
        self._layers = []

        for i, module_tuple in enumerate(self._network_architecture):

            if isinstance(net, tfd.Distribution):
                # net = net.sample(self.n_z_samples)
                net = net.sample()

            if len(module_tuple) == 3:
                module_name, module_kwargs, bool_activation = module_tuple
                # no decorators
                decorators = []
            elif len(module_tuple) == 4:
                module_name, module_kwargs, bool_activation, decorators = module_tuple
                # make a copy, so that I can pop safetely
                decorators = decorators.copy()
            else:
                raise Exception("The length of the module_tuple should be 3 or 4: " + str(module_tuple))

            if module_name in self._default_modules_kwargs:
                kwargs = {**self._default_modules_kwargs[module_name],
                          **module_kwargs
                          }

            elif module_name in self._default_layers_kwargs:
                kwargs = {**self._default_layers_kwargs[module_name],
                          **module_kwargs
                          }
            else:
                raise Exception(module_name + " is neither a sonnet module nor a tf layer. Hint: check if you have set the self._default_modules_kwargs or self._default_layers_kwargs")

            # add reference for the contractive regularizers if needed,
            # this works for both StochaticLayers and also or regular layers + a ConstractiveRegularizer decorator
            self.add_reference_for_contractive_regularizers(kwargs, inputs)

            # load decorators
            decorator_modules = []
            while len(decorators) > 0:
                decorator_name = decorators.pop(0)
                decorator = load_sonnet_module(decorator_name, kwargs)  # (sntmodule)
                decorator_modules.append(decorator)
                # linear = decorator(args)(snt.Linear)(flat)

            if len(decorator_modules) > 0:
                # only load class of the moduke without instantiation
                sntmodule = load_sonnet_module(module_name, kwargs, instantiate=False)
                while len(decorator_modules) > 0:
                    decorator = decorator_modules.pop(-1)
                    sntmodule = decorator(sntmodule)(kwargs)
                    # decorator(snt.Linear)(kwargs)(net)
            else:
                # no decorators, load the module
                sntmodule = load_sonnet_module(module_name, kwargs)

            # give the name feature to the last layer (after activation)
            last_node_before_logits = net
            net = sntmodule(net)

            self._modules.append(sntmodule)

            # at this point net is expected to be either a tensor (logits) or a tf.Distribution
            self._layers.append(net)

            if bool_activation:
                if isinstance(net, tfd.Distribution):
                    raise Exception("cannot apply activation to a tf.Distribution, check network architecture!")

                net = self._activation(net)

        # name the last layer before logits (or distribution)
        last_node_before_logits = tf.identity(last_node_before_logits, name="features")

        # this will be either a tensor or a distribution
        output = net

        return output

    def add_reference_for_contractive_regularizers(self, kwargs, reference_node):
        if "contractive_regularizer" in kwargs and kwargs["contractive_regularizer"] is not None:
            reg_name, reg_kwargs = kwargs["contractive_regularizer"]
            kwargs["contractive_regularizer"] = (reg_name, reg_kwargs, reference_node, self._network_str)

# #TODO-ARGO2 no need to have this as a separate function? GaussianModelLatentVars should be a Sonnet Module
# #TODO-ARGO2 also: how do we plan on attaching several stochastic layers in the default way (one can always rewrite completely the network from scratch and i would say it is usually recommended.) ?
# def _create_stochastic_layer(self, layer, stochastic_architecture, inputs, n_samples):
#     # NP: multiple layers not implented yet, in case you would need
#     #     a loop over self.network_architecture["stochastic"]
#     assert(len(stochastic_architecture)==1)
#
#     stochastic_model = None
#     layer_tuple = stochastic_architecture[0]
#
#     layer_name, layer_kwargs = layer_tuple
#
#     kwargs = {
#         **self._default_layers_kwargs["stochastic"],
#         **layer_kwargs,
#         "name" : "stochastic_layer"
#     }
#
#     try:
#         # local import
#         # core_module = importlib.import_module("core")
#         # layer_path = layer_name+'.'+layer_name
#         # layer_class = load_method_fn_from_method_path(core_module, layer_path)
#
#         #TODO-ARGO2 we NEED to put stochastic layers in areasonable place
#         layer_module = importlib.import_module("."+layer_name, '.'.join(__name__.split('.')[:-1]))
#         layer_class = load_method_fn_from_method_path(layer_module, layer_name)
#
#         #TODO-ARGO2 having dictionary of parameters in the init or in a method is discouraged.
#         #TODO-ARGO2 key arguments should be specified (as in all python and tf methods). See a usage example in utils.load_method_from_tuple
#         stochastic_model = layer_class(kwargs, n_samples)
#     except:
#         raise Exception("Invalid type of stochastic layer: " + layer_name)
#
#     # TODO-ARGO2 oh here? Not in the __init__ of the layer? the problem was the x.. then i think it is more clean should be passed..or?
#     if kwargs["contractive_regularizer"]:
#         # the x is passed as a parameter in case a contractive regularization is created
#         stochastic_model.initialize_contractive_regularizer(self.x)
#
#     # create the graph associated to the layer
#     stochastic_model.create_stochastic_layer(layer)
#     #TODO-ARGO2 can we use directly tf.distributions classes?
#
#     return stochastic_model
