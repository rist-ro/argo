import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import importlib
import re

import pdb

import tensorflow as tf
import sonnet as snt

from .argoLogging import get_logger
tf_logging = get_logger()

NUMTOL=1e-7 # NB if you plan to changee this, talk to me (Luigi)

AC_REGULARIZATION = "activity_and_contractive_regularizers"

def my_loss_full_logits(y, logits):
    
    n = logits.get_shape().as_list()[1]
    probabilities = tf.nn.softmax(logits)

    loss = tf.reduce_sum(-tf.one_hot(y,depth=n)*tf.log(probabilities + NUMTOL),axis=1)
    return loss

        
def make_list(l):
    return l if isinstance(l, list) else [l]

def create_list_colors(max_colors):
    r = max_colors/10.0
    #pdb.set_trace()
    return plt.cm.tab10((1/r*np.arange(10*r)).astype(int))

def load_sonnet_module(module_name, kwargs):
    
    tf_logging.info("Creating sonnet module " + str(module_name))
    
    try:
        # try first to get the module from argo
        layer_module = importlib.import_module("." + module_name, '.'.join(__name__.split('.')[:-1]))
        sntmodule = eval_method_from_tuple(layer_module, (module_name, kwargs))
    except ImportError:
        # otherwise get module from sonnet or sonnet.nets or raise exception
        module = None
        if hasattr(snt, module_name):
            module = snt
        elif hasattr(snt.nets, module_name):
            module = snt.nets
        else:
            raise Exception("sonnet module " + module_name + " not recognized")
        sntmodule = eval_method_from_tuple(module, (module_name, kwargs))
        
    except Exception as e:

        raise Exception("problem loading module: %s, kwargs: %s, exception: %s" % (module_name, kwargs, e)) from e
    
    return sntmodule

def get_ac_collection_name(additional_str):
    ac_collection_name = AC_REGULARIZATION
    if additional_str:
        ac_collection_name += "_"+additional_str
    return ac_collection_name

def compose_name(basename, dataset_str):
    if basename[0] == "-":
        basename = basename[1:]
    return basename + "_" + dataset_str

# from https://stackoverflow.com/questions/47709854/how-to-get-covariance-matrix-in-tensorflow?rq=1
# once it is compatible with Python3, we should move to
# https://www.tensorflow.org/tfx/transform/api_docs/python/tft/covariance

# NB I cannot get the value of n_points, since I concatenated the tensors, thus I need to
# return the matrix up to the moltiplication by n_points. Annoying, but I cannot find a solution
def tf_cov_times_n_points(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    sum_x = tf.reduce_sum(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), sum_x)
    #n_points = x.shape.as_list()[0]
    vx = tf.matmul(tf.transpose(x), x) #/n_points
    cov_xx = vx - mx
    
    return cov_xx

def create_reset_metric(metric, scope, **metric_args):
    """create a metric inside a scope to control over variables reset operations.
    suggestion by: shoeffner -> https://github.com/tensorflow/tensorflow/issues/4814
    reimplemented and added check if scope is not empty, to avoid accidentally resetting some other variables.

    Args:
        metric (type): a tf.metric function, this typically returns a tuple: (metric_op, update_op).
        scope (type): scope_name to use in the creation of the metric nodes.
                        (scope should be different from any other scope already containing variables)
        **metric_args (type): arguments to pass to the metric function -> metric(**metric_args).

    Returns:
        (metric_op, update_op, reset_op)

    Example usage:
    ```python
        metric_op, update_op, reset_op = create_reset_metric(tf.metrics.mean,
                                                            scope="mean_reset_metric/"+tensor.name,
                                                            values=tensor)
    ```
    """

    scope = scope.replace(":","_")
    with tf.variable_scope(scope) as scope:
        local_vars = tf.contrib.framework.get_variables(scope,
                                    collection=tf.GraphKeys.LOCAL_VARIABLES)
        # this performs a check that the scope is currently empty,
        # this is very important to ensure the reset_op will reset
        # only the metric variables created in the present function
        if local_vars:
            raise Exception("local variables already present in scope: `%s`. "\
                            "I cannot safely initialize reset operation for the metric." % scope.name)
        
        metric_op, update_op = metric(**metric_args)
        local_vars = tf.contrib.framework.get_variables(scope,
                                    collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(local_vars)

    return metric_op, update_op, reset_op


def create_concat_opts(scope, node):
    #self.concat_ops[ds_key] = tf.contrib.framework.get_variables(scope,
    #                                                             collection=tf.GraphKeys.LOCAL_VARIABLES)
    #if self.concat_ops[ds_key]:
    #    raise Exception("variable already present in scope: `%s`. "\
    #                    "I cannot safely initialize reset operation for the metric." % scope.name)

    scope = scope.replace(":","_")
    
    with tf.variable_scope(scope) as scope:
        #local_vars = tf.contrib.framework.get_variables(scope,
        #                                       collection=tf.GraphKeys.LOCAL_VARIABLES)
        #if local_vars:
        #    raise Exception("local variables already present in scope: `%s`. "\
        #                    "I cannot safely initialize reset operation for the metric." % scope.name)
        
        # see https://github.com/tensorflow/tensorflow/issues/4432

        #TODO it must be 1-D? Maybe yes,(for PCA yes for sure).
        node_shape = node.shape.as_list()
        if len(node_shape)!=2:
            raise RuntimeError("the node passed for concatenation is not a 2D tensor, as expected...")
        
        dim = node_shape[1]
        
        accumulator = tf.get_variable("accumulator",
                                      initializer = tf.zeros([0, dim]),
                                      trainable = False,
                                      collections = [tf.GraphKeys.LOCAL_VARIABLES]
                                      )

        i = tf.get_variable("index",
                            initializer = tf.constant(0),
                            dtype = tf.int32,
                            trainable = False,
                            collections = [tf.GraphKeys.LOCAL_VARIABLES]
                            )
        
        def assign():
            #with tf.control_dependencies(tf.assign_add(i, 1)):
            return tf.assign(accumulator, node, validate_shape=False), tf.assign_add(i, 1)

        def concat():
            return tf.assign(accumulator, tf.concat([accumulator, node], axis=0), validate_shape=False), tf.assign_add(i, 1)
        
        concat_update_ops = tf.cond(tf.equal(i, 0),
                                    assign,
                                    concat)
                 
        concat_reset_ops = tf.variables_initializer([i])

        return accumulator, concat_update_ops, concat_reset_ops


def sample_discrete_from_continuous(probabilies):
    bernoulli = tf.distributions.Bernoulli(probs = probabilies)
    return bernoulli.sample()

def tf_sample_discrete_from_continuous(X):
    return np.random.binomial(1, X, size=X.shape)

def add_gaussian_noise_and_clip(data, variance=0.1, low=0., high=1.):
    noise = np.random.normal(0, variance, size=X.shape)
    noisy_data = data + noise
    # clip in [low,high]
    noisy_data_clipped = np.clip(noisy_data, low, high)
    return noisy_X_clipped, noise

def tf_add_gaussian_noise_and_clip(data, variance=0.1, low=0., high=1.):
    noise = tf.distributions.Normal(tf.zeros_like(data), tf.ones_like(data)*np.sqrt(variance)).sample()
    noisy_data = data + noise
    # clip in [low,high]
    noisy_data_clipped = tf.clip_by_value(noisy_data, low, high)
    return noisy_data_clipped, noise


def rescale(data, eps, min_value=0., max_value=1.):
    delta = max_value - min_value
    return (delta - 2*eps) * data + eps + min_value

def tf_rescale(data, eps, min_value=0., max_value=1.):
    delta = max_value - min_value
    return (delta - 2*eps) * data + eps + min_value

def clip(data, low=0., high=1.):
    return np.clip(data, low, high)

def tf_clip(data, low=0., high=1.):
    return tf.clip_by_value(data, low, high)


def np_softplus(x, limit=30):
      if x>limit:
          return x
      else:
          return np.log(1.0 + np.exp(x))

dtype_short={'float16':'f16','float32':'f32','float64':'f64','bfloat16':'bf16','complex64':'c64','complex128':'c128'}

def get_short_dtype(dtypestr):
    """
    return the dtype name (string) in short form, typically used for id construction

    Args:
        dtypestr (str) : dtype name in string format

    Returns:
        str : the short name
    """
    if not dtypestr in dtype_short:
        raise ValueError('the type specified %s is not supported.'%dtypestr)
    return dtype_short[dtypestr]


# layer_short_names={'dense' : 'D',
#                     'conv2d' : 'C',
#                     'Linear' : 'D',
#                     'Conv2D' : 'C',
#                    'GaussianDiagonal' : 'GD',
#                    'GaussianDiagonalZeroOne' : 'GDZO',
#                    'LogitNormalDiagonal' : 'LND',
#                    'Bernoulli' : 'B'}
#
# def get_short_layer_name(layer_str, layer_kwargs):
#     """
#     return the layer type name (string) in short form, typically used for id construction
#
#     Args:
#         layer_str (str) : layer type in string format
#
#     Returns:
#         str : the short name
#     """
#     if not layer_str in layer_short_names:
#         raise ValueError('the type specified %s is not supported.'%layer_str)
#     return layer_short_names[layer_str]
#
#

regularizers_short_names={'standard_contractive_regularizer' : 'scr',
                          'geometric_contractive_regularizer' : 'gcr'}


def get_short_regularization_name(reg_name):
    """
    return the regularization name (string) in short form, typically used for id construction

    Args:
        reg_name (str) : regularization name in string format

    Returns:
        str : the short name
    """
    if not reg_name in regularizers_short_names:
        raise ValueError('the regularizer specified %s is not supported.'%reg_name)
    return regularizers_short_names[reg_name]



def regularization_info(layer_dict):
    #"contractive_regularizer" : ("standard_contractive_regularizer",
                # {"norm": 2, "scale_mean" : 0.1, "scale_covariance" : 0.1})
    reg_info=""
    contr_reg = layer_dict.get("contractive_regularizer", None)
    if contr_reg is not None:
        crname, crdict = contr_reg
        reg_info += get_short_regularization_name(crname)
        reg_info += "n"+str(crdict["norm"])+"sm"+str(crdict["scale_mean"])+"sc"+str(crdict["scale_covariance"])

    # TODO add your own regularizer type and extract relevant parameters!

    return reg_info


method_name_short = {
#activation functions
'relu':'R',
'elu':'E',
'leaky_relu':'LR',
'sigmoid':'S',
'tanh':'T',    

#layers
'dense':'D',
'conv2d':'C',
'max_pooling2d':'P',
'flatten':'', # not shown

# snt modules
'Linear':'D',
'LinearWN':'D',
'Conv2D':'C',
'Conv2DTranspose':'CT',
'Conv2DWN':'C',
'ConvNet2D':'CN',
'ConvNet2DTranspose':'CNT',
'BatchFlatten':'',
'BatchNorm':'BN',
'BatchReshape':'BR',
'ResUnit':'RU',
'ResNet18':'RN18',
'Sigmoid':'S',
    
# stochastic_models
'GaussianDiagonal' : 'GD',
'GaussianDiagonalZeroOne' : 'GD01',
'LogisticDiagonalZeroOne' : 'LD01',
'LogitNormalDiagonal' : 'LND',
'Bernoulli' : 'B',

# initializers
"xavier_initializer" : 'x',
"truncated_normal_initializer" : 't',
"constant_initializer" : 'c',

# custom networks
"CIFAR10Tutorial" : "CIFAR10Tutorial",
    
# regularizers
"l2_regularizer" : "Ltwo",
"l1_regularizer" : "Lone",

# covariance parameterization
"softplus" : 'S',
"exp" : 'E',

# zero one methods
"clip" : 'c',
"sigmoid" : 's',

# clipping gradient
"clip_by_global_norm" : "GN",
"clip_by_norm" : "N",
"clip_by_value" : "V",
"none" : "No",
}

def get_method_id(method_tuple):
    """Creates the id for a method of tensorflow.

    Args:
        method_tuple (tuple): A tuple composed of : (name of the method of tensorflow, kwargs to pass to the method, bool_activation).

    Returns:
        string: the idname of the method that we want to concatenate in the output filenames.

    """
    #ipdb.set_trace()
    #the name could be articulated, since I might want to get the initializers or regularizers
    #from different submodules in tf.
    #e.g. tf.contrib.layers.xavier_initializer
    #the method name I am interested in is the one after the last dot.

    listWithPoints = lambda x: "-".join(re.sub('[( )\[\]]', '',str(x)).replace(' ', '').split(","))

    method_name = method_tuple[0].split('.')[-1]
    method_kwargs = method_tuple[1]

    methodid = method_name_short[method_name]

    if method_name=='dense':
        methodid += str(method_kwargs['units'])

    elif method_name=='conv2d':
        methodid += str(method_kwargs['filters'])+'x'+re.sub('[( )]', '', str(method_kwargs['kernel_size']))

    elif method_name=='max_pooling2d':
        methodid += re.sub('[( )]', '', str(method_kwargs['pool_size']))

    elif method_name=='flatten':
        pass

    elif method_name=='Linear':
        methodid += str(method_kwargs['output_size'])

    elif method_name=='LinearWN':
        methodid += str(method_kwargs['output_size']) + 'wn'+str(int(method_kwargs['use_weight_norm']))

    elif method_name=='Conv2D':
        methodid += str(method_kwargs['output_channels'])+'o'+listWithPoints(method_kwargs['kernel_shape'])

    elif method_name=='Conv2DTranspose':
        methodid += str(method_kwargs['output_channels'])+'o'+listWithPoints(method_kwargs['output_shape'])+'k'+listWithPoints(method_kwargs['kernel_shape'])
        
    elif method_name=='Conv2DWN':
        methodid += str(method_kwargs['output_channels'])+'o'+listWithPoints(method_kwargs['kernel_shape'])+\
                        'wn'+str(int(method_kwargs['use_weight_norm']))
    
    elif method_name=='ResUnit':
        #TODO
        pass
       
    elif method_name=='ResNet18':
        methodid += '-o'+str(method_kwargs['output_size'])+'wn'+str(int(method_kwargs['use_weight_norm']))
        
    elif method_name=='ConvNet2D':
        methodid += '-o' + listWithPoints(method_kwargs['output_channels']) + 'k' + listWithPoints(
            method_kwargs['kernel_shapes']) + 's' + listWithPoints(method_kwargs['strides'])

    elif method_name=='ConvNet2DTranspose':
        methodid += '-o' + listWithPoints(method_kwargs['output_channels']) + 'k' + listWithPoints(
            method_kwargs['kernel_shapes']) + 's' + listWithPoints(method_kwargs['strides'])

    elif method_name=='BatchFlatten':
        pass

    elif method_name=='Sigmoid':
        pass
    
    elif method_name=='BatchReshape':
        pass

    elif method_name=='BatchNorm':
        methodid += '' #+ str(method_kwargs['offset']) + 's' + str(method_kwargs['scale']) + 'd' + str(method_kwargs['decay_rate'])
    
    elif method_name=='GaussianDiagonal' \
            or method_name=='GaussianDiagonalZeroOne' \
            or method_name=='LogitNormalDiagonal' \
            or method_name=='LogisticDiagonalZeroOne':
        wrapped_module_name, wrapped_module_kwargs = method_kwargs["module_tuple"]
        # LUIGI the lower case is done to increase readibility
        methodid += "" + method_name_short[wrapped_module_name].lower()
        if "output_size" in method_kwargs:
            methodid += str(method_kwargs["output_size"])
        if "minimal_covariance" in method_kwargs and method_kwargs["minimal_covariance"] != 0:
            methodid += ".mc" + str(method_kwargs["minimal_covariance"])
        if "zero_one_method" in method_kwargs and method_kwargs["zero_one_method"] != "sigmoid":
            methodid += ".zo" + str(method_name_short[method_kwargs["zero_one_method"]])
        if "scalar_covariance" in method_kwargs:
            scov = method_kwargs["scalar_covariance"]
            if scov==True:
                methodid += ".scT"
            elif isinstance(scov, float):
                methodid += ".sc"+str(scov)
        
        methodid += regularization_info(method_kwargs)
    
    elif method_name=='Bernoulli':
        if "output_size" in method_kwargs:
            methodid += str(method_kwargs["output_size"])

    elif method_name=='CIFAR10Tutorial':
        pass
            
    elif "xavier_initializer" in method_name:
        pass

    elif "truncated_normal_initializer" in method_name:
        methodid += str(method_kwargs['stddev'])
    
    elif "constant_initializer" in method_name:
        methodid += str(method_kwargs['value'])
    
    elif "l1_regularizer" in method_name or "l2_regularizer" in method_name:
        methodid += str(method_kwargs["scale"])

    elif method_name == "softplus":
        pass

    elif method_name == "exp":
        pass

    #Here implement your favourite method
    #elif      :
    #

    else:
        print('----------------------')
        print('ERROR ', method_name)
        raise ValueError("id rule for method `%s` has to be implemented." % method_name)

    return methodid

def get_clipping_id(clipping_tuple):
    method = clipping_tuple[0]
    if not method:
        return method_name_short["none"]
    else:
        value = clipping_tuple[1]["value"]
        return method_name_short[method] + str(value)

def eval_method_from_tuple(module, method_tuple):
    """

    Args:
        module (python module): module from which take the method.
        method_tuple (tuple): (method_path, method_kwargs).

    Returns:
        if method_tuple is None returns None
        otherwise returns
        module.method_path(**method_kwargs)

    """

    if not method_tuple:
        return None

    method_fn = load_method_fn_from_method_path(module, method_tuple[0])

    return method_fn(**method_tuple[1])


def load_method_fn_from_method_path(module, method_path):
    """

    Args:
        module (python module): module from which take the method.
        method_path (string): path to the method.

    Returns:
        if method_path is None returns None
        otherwise returns
        module.method_path(**method_kwargs)

    """

    if not method_path:
        return None

    mpathsplit = method_path.split(".")
    method_name = mpathsplit[-1]

    path = module.__name__
    middle_path = '.'.join(mpathsplit[:-1])
    if middle_path:
        path += '.' + middle_path

    last_module = importlib.import_module(path)
    method_fn = getattr(last_module, method_name)
    return method_fn


def load_class(module_plus_class, relative=False, base_path=''):
    # assemble class path
    class_path = ''
    #if class_base_path prepend this to the path
    if base_path:
        class_path = base_path
        #if the prepended path does not finish with a dot add it
        if class_path[-1] != '.':
            class_path += '.'
    class_path += module_plus_class

    #split in all modules and class
    modulesname, classname = class_path.rsplit('.', 1)
    if relative:
        modulesname = class_path.split('.', 1)[1]

    my_module = importlib.import_module(modulesname)
    my_class = getattr(my_module, classname)
    return my_class

def eval_file(file_name_path):
    with open(file_name_path, 'r') as fstream:
        return eval(fstream.read())
