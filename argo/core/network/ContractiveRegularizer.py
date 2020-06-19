import pdb
import importlib

import tensorflow as tf

import copy

import numbers

from ..utils.argo_utils import NUMTOL, get_ac_collection_name, eval_method_from_tuple

from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

class ContractiveRegularizer(object):
    
    def __init__(self, **decor_arg):    

        print("ConstractiveRegularizedModule")
               
        #string = string
        #print("Object created:" + string)
        
        #self._other_class = cls

        #pdb.set_trace()
        
        assert("contractive_regularizer" in decor_arg)

        contractive_regularizer = decor_arg.pop("contractive_regularizer")

        '''
        # manual copy not needed
        decor_arg_minus_contractive = {}
        for k, d in decor_arg.items():
            if k != "contractive_regularizer":
                decor_arg_minus_contractive[k] = d
        '''
        
        self._contractive_regularizer = contractive_regularizer
        self._decor_arg = decor_arg #_minus_contractive
       
        '''
        @property
        def test_set_x_fileName(self):
            return self._test_set_x_fileName
        
        @test_set_x_fileName.setter
        def test_set_x_fileName(self, test_set_x_fileName):
            self._test_set_x_fileName = test_set_x_fileName
        '''
        
    def __call__(self, cls):

        print("def __call__(self, cls)" + str(cls))
        #self.snt_module = cls

        # args of the sonnet module
        decor_arg = self._decor_arg
        contractive_regularizer = self._contractive_regularizer
                    
        class ConstractiveRegularizedModule(cls):
            #print("class ConstractiveRegularizedModule(cls)" + str(cls))

            '''
            def __init__(self, cls):
                self._contractive_regularizer_tuple = None
                self._contractive_regularizer_inputs = None
                self._contractive_collection_network_str = ""
            '''
            
            #pdb.set_trace()
            #snt_module = cls

            #self._decor_arg = self._decor_arg
                        
            def __call__(self, *cls_args):
                # cls_args is the node to which I attach _build
                
                # calling _build of the sonnet module
                snt_layer = cls(**decor_arg)(*cls_args)

                contractive_regularizer_name, \
                    contractive_regularizer_kwargs, \
                    contractive_regularizer_inputs,\
                    contractive_collection_network_str = contractive_regularizer
            
                contractive_regularizer_tuple = (contractive_regularizer_name,
                                                 contractive_regularizer_kwargs)

                self.set_contractive_regularizer(snt_layer,
                                                 contractive_regularizer_inputs,
                                                 contractive_regularizer_tuple,
                                                 contractive_collection_network_str)

                return snt_layer

            @property
            def _contractive_regularizer_filename(self):
                return ".ContractiveRegularizer"
    
            def set_contractive_regularizer(self, node, x, contractive_regularizer_tuple = None, contractive_collection_network_str=""):
                if contractive_regularizer_tuple:
                    regularizer_name, regularizer_kwargs = contractive_regularizer_tuple
                    try:
                        reg_module = importlib.import_module(self._contractive_regularizer_filename, '.'.join(__name__.split('.')[:-1]))
                        contractive_regularizer = eval_method_from_tuple(reg_module, (regularizer_name, regularizer_kwargs))
                        #pdb.set_trace()
                    except ImportError as e:
                        raise ImportError("regularizer %s not found" % regularizer_name) from e
            
                    reg_node = contractive_regularizer(node, x)
                    
                    ac_collection_name = get_ac_collection_name(contractive_collection_network_str)
                    tf.add_to_collection(ac_collection_name, reg_node)
        return ConstractiveRegularizedModule

def standard_contractive_regularizer(scale, norm, trick=False, scope=None):
    # Returns a dictionary of functions that can be used to compute the regularizations
    # to the node

    # x is input to the network

    # input validation
    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                             scale)
        #if scale == 0.:
        #    return lambda _: None

    use_pfor = False
    def contractive_regularizer(node, x, reg_name="standard_contractive_regularizer"):
        with tf.name_scope(scope, reg_name, [node, x]) as name:
            scale_regularizer = tf.convert_to_tensor(scale,
                                                     dtype=node.dtype.base_dtype,
                                                     name='scale')

            norm_jac = tf.norm(batch_jacobian(node, x, use_pfor = use_pfor), ord = norm, name = "norm_jac")

            reg_node = tf.multiply(scale_regularizer,
                                   norm_jac,
                                   name=reg_name)

            #tf.summary.scalar(norm_jac.name, norm_jac)

            return reg_node

    return contractive_regularizer
