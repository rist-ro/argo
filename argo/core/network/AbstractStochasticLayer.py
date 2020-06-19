import tensorflow as tf
import sonnet as snt
import numpy as np

from operator import xor
import importlib

from ..utils.argo_utils import load_sonnet_module

import pdb

from abc import ABC, abstractmethod
from tensorflow_probability import distributions as tfd

from .AbstractModule import AbstractModule

from ..utils.argo_utils import NUMTOL, get_ac_collection_name, eval_method_from_tuple

# CONST = "constant"

class AbstractStochasticLayer(AbstractModule):
    
    def __init__(self, module_tuple = ("Linear", {}),
                 output_size=None,
                 output_shape=None,
                 minimal_covariance=0,
                 covariance_parameterization="softplus",
                 scalar_covariance = False,
                 vectorial_covariance = False,
                 initializers = {},
                 regularizers = {},
                 contractive_regularizer = None,
                 name='abstract_stochastic'):
        """

        Args:
            either output_size or output_shape for the dimensions of the output layer.
            module_tuple : (module_name, module_kwargs), you can use Linear, LinearWN, (Conv2D, Conv2DWN,) Conv2DTranspose
                        if shape is not specified in the module_kwargs, it will be inferred using output_size and output_shape
                        of the AbstractGaussian module.
            minimal_covariance (float): covariance minimal threshold (in addition to NUMTOL, that is always present).
            covariance_parameterization (str): exp or softplus.
            scalar_covariance : one of [True, False, float]. If True or float a single scalar covariance will be used for all the distributions.
                                If a float is passed, will instantiate a constant scalar instead of a trainable variable. NB: please be aware that
                                covariance_parameterization is always applied on top of the initial node to get the covariance, this ensure to have
                                a positive value always, removing checks.
            initializers (dict): sonnet style initializers, dict of 'w' and 'b'.
            regularizers (dict): sonnet style regularizers, dict of 'w' and 'b'.
            contractive_regularizer (tuple): (reg_name, reg_kwargs, ref_node, network_str). network_str is for collection to which add the regularization.
            name (str): name of this sonnet module.
        
        """

        super().__init__(name = name)

        if (output_size is not None) and (output_shape is not None):
            raise ValueError("Either output_size or output_shape mut be specified, not both")
        
        self._output_shape = None
        
        if output_size is not None:
            self._output_shape = [output_size]
        elif output_shape is not None:
            self._output_shape = output_shape
        
#         import pdb;pdb.set_trace()
        self._module_tuple = module_tuple
        self._extra_kwargs = {"initializers" : initializers,
                              "regularizers" : regularizers}
        
        self._cov_parameterization = covariance_parameterization
        
        self._contractive_regularizer_tuple = None
        self._contractive_regularizer_inputs = None
        self._contractive_collection_network_str = ""

        # check on the parameters
        assert(not (scalar_covariance!=False and vectorial_covariance!=False))
            
        self._scalar_constant = None
        if isinstance(scalar_covariance, bool):
            self._scalar_bool = scalar_covariance
        else:
            self._scalar_bool = True
            self._scalar_constant = float(scalar_covariance)

        self._vectorial_bool = vectorial_covariance
        #if isinstance(scalar_covariance, bool):
        #    self._vectorial_bool = vectorial_covariance
        #else:
        #    self._scalar_bool = True
        #    self._scalar_constant = float(scalar_covariance)

            
        if contractive_regularizer:
            contractive_regularizer_name, \
                contractive_regularizer_kwargs, \
                contractive_regularizer_inputs,\
                contractive_collection_network_str = contractive_regularizer
            
            self._contractive_regularizer_tuple = (contractive_regularizer_name,
                                                   contractive_regularizer_kwargs)
            
            self._contractive_regularizer_inputs = contractive_regularizer_inputs
            self._contractive_collection_network_str = contractive_collection_network_str
        
        self._minimal_covariance = minimal_covariance
    
    @property
    @abstractmethod
    def _contractive_regularizer_filename(self):
        pass
    
    def set_contractive_regularizer(self, mean, cov, x, contractive_regularizer_tuple = None, contractive_collection_network_str=""):
#         pdb.set_trace()
        if contractive_regularizer_tuple:
            regularizer_name, regularizer_kwargs = contractive_regularizer_tuple
            try:
                reg_module = importlib.import_module(self._contractive_regularizer_filename, '.'.join(__name__.split('.')[:-1]))
                contractive_regularizer = eval_method_from_tuple(reg_module, (regularizer_name, regularizer_kwargs))
#                 pdb.set_trace()
            except ImportError as e:
                raise ImportError("regularizer %s not found" % regularizer_name) from e
                
                
            reg_node = contractive_regularizer(mean, cov, x)
            if not isinstance(reg_node, list):
                 reg_node = [reg_node]
            
            ac_collection_name = get_ac_collection_name(contractive_collection_network_str)
            for r in reg_node:
                tf.add_to_collection(ac_collection_name, r)
    
    
    def _get_stride(self, in_shape, out_shape):
        reminders = [i%o for i,o in zip(in_shape[0:2], out_shape[0:2])]
        if np.any(reminders):
            raise Exception("the output_shape specified `%s` is not allowed since does not divide input_shape `%s`"%(out_shape[0:2], in_shape[0:2]))
        
        stride = [int(i/o) for i,o in zip(in_shape[0:2], out_shape[0:2])]
        return stride
    
    
    def check_key_not_in(self, key_name, kwargs, accepted_value=None):
        if key_name in kwargs.keys():
            if accepted_value and kwargs[key_name]==accepted_value:
                return
            
            raise Exception("you specified both %s `%s` in submodule and outputshape `%s`\
                            in the StochasticLayer wrapper. disallowed"%(key_name, kwargs[key_name], self._output_shape))
    
    def check_key_in(self, key_name, kwargs, kwargs_name, for_this_reason):
        if key_name not in kwargs.keys():
            raise Exception("key %s is missing in %s, it is required %s."%(key_name, kwargs_name, for_this_reason))    
    
    def force_between_zero_and_one(self, tnsr, method="sigmoid"):
        """Force the tensor in between zero and one.

        Args:
            tnsr (type): the tensor.
            method (type): sigmoid or clip.

        Returns:
            the transformed tensor with values in [0,1]
        """
        choices = ["clip", "sigmoid"]
        if method not in choices:
            return Exception("method `%s` not recognized, must be one of `%s`"%(method, choices))
        
        if method == "clip":
            tnsr = tf.clip_by_value(tnsr, 0., 1.)
        elif method == "sigmoid":
            tnsr = tf.sigmoid(tnsr)
        
        return tnsr
    
    def force_between_plus_minus_one(self, tnsr, method="tanh"):
        """Force the tensor in between zero and one.

        Args:
            tnsr (type): the tensor.
            method (type): sigmoid or clip.

        Returns:
            the transformed tensor with values in [0,1]
        """
        choices = ["clip", "tanh"]
        if method not in choices:
            return Exception("method `%s` not recognized, must be one of `%s`"%(method, choices))
        
        if method == "clip":
            tnsr = tf.clip_by_value(tnsr, -1., 1.)
        elif method == "tanh":
            tnsr = tf.tanh(tnsr)
        
        return tnsr

# TODO nobody seems to use this, remove..
    # def get_covariance_from_parameters_python(self, parameters):
    #     if self._cov_parameterization=='exp':
    #         covariance = np.exp(parameters) + self._minimal_covariance + NUMTOL
    #     elif self._cov_parameterization=='softplus':
    #         covariance = np_softplus(parameters)**2 + self._minimal_covariance + NUMTOL
    #     else:
    #         raise Exception("Invalid parameterization method for the covariance diagonal elements")
    #     return covariance

    def get_covariance_from_parameters_tf(self, parameters):
        
        if self._cov_parameterization=='exp':
            covariance = tf.exp(parameters) + self._minimal_covariance + NUMTOL
        elif self._cov_parameterization=='softplus':
            covariance = tf.square(tf.nn.softplus(parameters)) + self._minimal_covariance + NUMTOL
        elif self._cov_parameterization=='linear_softplus':
            covariance = tf.nn.softplus(parameters) + self._minimal_covariance + NUMTOL
        else:
            raise Exception("Invalid parameterization method for the covariance diagonal elements: " + self._cov_parameterization)
        return covariance

    def prepare_input_and_merge_module_kwargs(self, inputs, module_name, module_kwargs):

        extra_kwargs = self._extra_kwargs
        # if self._output_shape is set I should write sizes in the Submodule
        if module_name == "Linear" or module_name == "LinearWN":
            inputs = tf.layers.flatten(inputs)
            if self._output_shape is not None:
                self.check_key_not_in("output_size", module_kwargs)
                extra_kwargs["output_size"] = np.prod(self._output_shape)
            else:
                self._output_shape = [module_kwargs["output_size"]]

        elif module_name == "Conv2D" or module_name == "Conv2DWN":
            if self._output_shape is not None:
                self.check_key_not_in("output_channels", module_kwargs)
                self.check_key_not_in("stride", module_kwargs)
                self.check_key_not_in("padding", module_kwargs, 'SAME')

                extra_kwargs["output_channels"] = self._output_shape[2]
                extra_kwargs["stride"] = self._get_stride(inputs.shape.as_list()[1:],
                                                          self._output_shape)
        elif module_name == "Conv1D":
            if self._output_shape is not None:
                self.check_key_not_in("output_channels", module_kwargs)
                self.check_key_not_in("padding", module_kwargs, 'SAME')
                extra_kwargs["output_channels"] = self._output_shape[2]

        elif module_name == "Conv2DTranspose" or module_name == "Conv2DTransposeWN":
            if self._output_shape is not None:
                self.check_key_not_in("output_channels", module_kwargs)
                self.check_key_not_in("stride", module_kwargs)
                self.check_key_not_in("output_shape", module_kwargs)
                extra_kwargs["output_shape"] = self._output_shape[0:2]
                extra_kwargs["stride"] = self._get_stride(self._output_shape,
                                                          inputs.shape.as_list()[1:])
                extra_kwargs["output_channels"] = self._output_shape[2]

        else:
            raise Exception(
                "module name `%s` is not allowed, is not implemented yet to be wrapped by a Stochastic Layer." % module_name)

        kwargs = {**extra_kwargs,
                  **module_kwargs}

        return inputs, kwargs

    def create_mean_n_cov_layers(self, inputs):
        # create the layers for mean and covariance
#         import pdb;pdb.set_trace()
        module_name, module_kwargs = self._module_tuple
        inputs, kwargs = self.prepare_input_and_merge_module_kwargs(inputs, module_name, module_kwargs)

        sntmodule_mean = load_sonnet_module(module_name, kwargs)
        mean = sntmodule_mean(inputs)

        # decide how to instantiate the covariance parameters
        if not self._scalar_bool and not self._vectorial_bool:
            sntmodule_cov = load_sonnet_module(module_name, kwargs)
            covariance_params = sntmodule_cov(inputs)
        else:
            if self._scalar_bool:
                if self._scalar_constant is None:
                    self.check_key_in('b', self._extra_kwargs['initializers'], 'initializers',
                                      'to initialize the scalar covariance parameter')
                    init_cov = self._extra_kwargs['initializers']['b']
                    covariance_params = tf.get_variable(name="scalar_cov_param", shape=(), initializer=init_cov)
                else:
                    covariance_params = tf.constant(self._scalar_constant, name="scalar_cov_param")
            elif self._vectorial_bool:
                new_kwargs = kwargs.copy()
                new_kwargs["output_size"] = 1
                sntmodule_cov = load_sonnet_module(module_name, new_kwargs)
                covariance_params = sntmodule_cov(inputs)
            else:
                raise Exception("unexpected condition")
                    
        # if I am using linear layers I need to reshape to match output_shape
        if module_name == "Linear" or module_name == "LinearWN":
            output_shape = [-1] + self._output_shape
            mean = tf.reshape(mean, output_shape)
            if not self._scalar_bool and not self._vectorial_bool:
                covariance_params = tf.reshape(covariance_params, output_shape)

        covariance = self.get_covariance_from_parameters_tf(covariance_params)
        # I need the standard deviation for the Normal tf distribution
        scale = tf.sqrt(covariance)

        return mean, covariance, scale
