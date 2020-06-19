import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from tensorflow.bitwise import bitwise_xor
from itertools import product

import hashlib

import numpy as np
import importlib
import re

import pdb

import tensorflow as tf
import sonnet as snt

from ..argoLogging import get_logger

import copy

tf_logging = get_logger()

NUMTOL = 1e-7  # NB if you plan to changee this, talk to me (Luigi)

AC_REGULARIZATION = "activity_and_contractive_regularizers"
CUSTOM_REGULARIZATION = "custom_regularizers"


def create_panels_lists(list_of_vpanels_of_plots):
    nodes_to_log = []
    names_of_nodes_to_log = []
    filenames_to_log_to = []

    for vpanel in list_of_vpanels_of_plots:
        nodes_vpanel = []
        names_vpanel = []
        files_vpanel = []
        for plot in vpanel:
            assert isinstance(plot["nodes"], list), "`nodes` in a plot dictionary must be a list"
            assert isinstance(plot["names"], list), "`names` in a plot dictionary must be a list"
            assert isinstance(plot["output"], dict), "`output` in a plot dictionary must be a dict"

            nodes_vpanel.append(plot["nodes"])
            names_vpanel.append(plot["names"])
            files_vpanel.append(plot["output"])

        nodes_to_log.append(nodes_vpanel)
        names_of_nodes_to_log.append(names_vpanel)
        filenames_to_log_to.append(files_vpanel)

    return nodes_to_log, names_of_nodes_to_log, filenames_to_log_to


def update_conf_with_defaults(opts, default_params):
    
    # update the defaul values in opts
    # use .update to modify the dict in place

    # originally
    #passed_opts = opts.copy()
    #opts.update(self.default_params)
    #opts.update(passed_opts)
    
    # new
    copy_opts = copy.deepcopy(opts)
    copy_opts.update(default_params)
    copy_opts.update(opts)
    
    return copy_opts


def my_loss_full_logits(y, logits):
    n = logits.get_shape().as_list()[1]
    probabilities = tf.nn.softmax(logits)

    loss = tf.reduce_sum(-tf.one_hot(y, depth=n) * tf.log(probabilities + NUMTOL), axis=1)
    return loss


def make_list(l):
    return l if isinstance(l, list) else [l]


def create_list_colors(max_colors):
    r = max_colors / 10.0
    return plt.cm.tab10((1 / r * np.arange(10 * r)).astype(int))


def load_sonnet_module(module_name, kwargs, instantiate=True):
    tf_logging.info("Loading sonnet module " + str(module_name))

    try:
        my_path = '.'.join(__name__.split('.')[:-2])
        # try first to get the module from argo
        layer_module = importlib.import_module(my_path + ".network." + module_name)
        sntmodule = eval_method_from_tuple(layer_module, (module_name, kwargs), instantiate)

    except ImportError:
        # otherwise get module from sonnet or sonnet.nets or raise exception
        module = None
        if hasattr(snt, module_name):
            module = snt
        elif hasattr(snt.nets, module_name):
            module = snt.nets
        else:
            raise Exception("sonnet module " + module_name + " not recognized")
        sntmodule = eval_method_from_tuple(module, (module_name, kwargs), instantiate)

    except Exception as e:
        raise Exception("problem loading module: %s, kwargs: %s, exception: %s" % (module_name, kwargs, e)) from e

    return sntmodule


def get_ac_collection_name(additional_str):
    ac_collection_name = AC_REGULARIZATION
    if additional_str:
        ac_collection_name += "_" + additional_str
    return ac_collection_name


def compose_name(basename, dataset_str, separator="_"):
    if basename[0] == "-":
        basename = basename[1:]
    return basename + separator + dataset_str


def hash_this(longstring, trunc=None):
    hasher = hashlib.sha1(longstring.encode('utf-8'))
    hexstr = hasher.hexdigest()
    if trunc:
        hexstr=hexstr[:trunc]

    return hexstr


# from https://stackoverflow.com/questions/47709854/how-to-get-covariance-matrix-in-tensorflow?rq=1
# once it is compatible with Python3, we should move to
# https://www.tensorflow.org/tfx/transform/api_docs/python/tft/covariance

# NB I cannot get the value of n_points, since I concatenated the tensors, thus I need to
# return the matrix up to the moltiplication by n_points. Annoying, but I cannot find a solution
def tf_cov_times_n_points(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    sum_x = tf.reduce_sum(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), sum_x)
    # n_points = x.shape.as_list()[0]
    vx = tf.matmul(tf.transpose(x), x)  # /n_points
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

    scope = scope.replace(":", "_")
    with tf.variable_scope(scope) as scope:
        local_vars = tf.contrib.framework.get_variables(scope,
                                                        collection=tf.GraphKeys.LOCAL_VARIABLES)
        # this performs a check that the scope is currently empty,
        # this is very important to ensure the reset_op will reset
        # only the metric variables created in the present function
        if local_vars:
            raise Exception("local variables already present in scope: `%s`. " \
                            "I cannot safely initialize reset operation for the metric." % scope.name)

        metric_op, update_op = metric(**metric_args)
        local_vars = tf.contrib.framework.get_variables(scope,
                                                        collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(local_vars)

    return metric_op, update_op, reset_op


def create_concat_opts(scope, node):
    # self.concat_ops[ds_key] = tf.contrib.framework.get_variables(scope,
    #                                                             collection=tf.GraphKeys.LOCAL_VARIABLES)
    # if self.concat_ops[ds_key]:
    #    raise Exception("variable already present in scope: `%s`. "\
    #                    "I cannot safely initialize reset operation for the metric." % scope.name)

    scope = scope.replace(":", "_")

    with tf.variable_scope(scope) as scope:
        # local_vars = tf.contrib.framework.get_variables(scope,
        #                                       collection=tf.GraphKeys.LOCAL_VARIABLES)
        # if local_vars:
        #    raise Exception("local variables already present in scope: `%s`. "\
        #                    "I cannot safely initialize reset operation for the metric." % scope.name)

        # see https://github.com/tensorflow/tensorflow/issues/4432

        # TODO it must be 1-D? Maybe yes,(for PCA yes for sure).
        node_shape = node.shape.as_list()
        if len(node_shape) != 2:
            raise RuntimeError("the node passed for concatenation is not a 2D tensor, as expected...")

        dim = node_shape[1]

        accumulator = tf.get_variable("accumulator",
                                      initializer=tf.zeros([0, dim]),
                                      trainable=False,
                                      collections=[tf.GraphKeys.LOCAL_VARIABLES]
                                      )

        i = tf.get_variable("index",
                            initializer=tf.constant(0),
                            dtype=tf.int32,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                            )

        def assign():
            # with tf.control_dependencies(tf.assign_add(i, 1)):
            return tf.assign(accumulator, node, validate_shape=False), tf.assign_add(i, 1)

        def concat():
            return tf.assign(accumulator, tf.concat([accumulator, node], axis=0), validate_shape=False), tf.assign_add(i, 1)

        concat_update_ops = tf.cond(tf.equal(i, 0),
                                    assign,
                                    concat)

        concat_reset_ops = tf.variables_initializer([i])

        return accumulator, concat_update_ops, concat_reset_ops

'''
def sample_discrete_from_continuous(probabilies):
    bernoulli = tf.distributions.Bernoulli(probs=probabilies)
    return bernoulli.sample()
'''

def tf_sample_discrete_from_continuous(X):
    raise Exception("Are you sure you want to use this with the new 0/1 encoding?")
    return tf.distributions.Bernoulli(probs=X, dtype=X.dtype).sample()

def tf_add_noise_to_discrete(X, flip_probability):
    probs = tf.ones(tf.shape(X)) * (1-flip_probability)
    flip = tf.distributions.Bernoulli(probs=probs, dtype=X.dtype).sample()
    #flipped = bitwise_xor(tf.cast(X, dtype=tf.int32), tf.cast(flip, dtype=tf.int32))
    flipped = X * (2*flip-1)
    return flipped

'''
def add_gaussian_noise_and_clip(data, variance=0.1, low=0., high=1.):
    noise = np.random.normal(0, variance, size=X.shape)
    noisy_data = data + noise
    # clip in [low,high]
    noisy_data_clipped = np.clip(noisy_data, low, high)
    return noisy_X_clipped, noise
'''

def tf_add_gaussian_noise_and_clip(data, std=0.1, low=-1., high=1., clip_bool=True):
    # fixed problem, here it was variance, but everybody thought it was std... std is better to control since we understand the 'range' of the noise
    noise = tf.distributions.Normal(tf.zeros_like(data), tf.ones_like(data) * std).sample()
    noisy_data = data + noise
    # clip in [low,high]
    if clip_bool:
        noisy_data = tf.clip_by_value(noisy_data, low, high)
    return noisy_data, noise

'''
def rescale(data, eps, min_value=0., max_value=1.):
    delta = max_value - min_value
    return (delta - 2 * eps) * data + eps + min_value
'''

def tf_rescale(data, eps, min_value=-1.0, max_value=1.0):
    delta = max_value - min_value
    return (delta - 2*eps)*data + eps + min_value

'''
def clip(data, low=-1, high=1):
    return np.clip(data, low, high)
'''

def tf_clip(data, low=-1.0, high=1.0):
    #data = tf.cast(data, dtype=tf.float32)
    return tf.clip_by_value(data, low, high)


def np_softplus(x, limit=30):
    if x > limit:
        return x
    else:
        return np.log(1.0 + np.exp(x))


dtype_short = {
    'float16':    'f16',
    'float32':    'f32',
    'float64':    'f64',
    'bfloat16':   'bf16',
    'complex64':  'c64',
    'complex128': 'c128'}


def get_short_dtype(dtypestr):
    """
    return the dtype name (string) in short form, typically used for id construction

    Args:
        dtypestr (str) : dtype name in string format

    Returns:
        str : the short name
    """
    if not dtypestr in dtype_short:
        raise ValueError('the type specified %s is not supported.' % dtypestr)
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

regularizers_short_names = {
    'standard_contractive_regularizer':  'SCR',
    'cos_contractive_regularizer':  'CCR',
    'geometric_contractive_regularizer': 'GCR',
    'wasserstein_contractive_regularizer': 'WCR',
    'ring_loss_regularizer': 'RLR',
    'ring_loss_variable_regularizer': 'RLVR',
    'contractive_reg_list': ''}


def get_short_regularization_name(reg_name):
    """
    return the regularization name (string) in short form, typically used for id construction

    Args:
        reg_name (str) : regularization name in string format

    Returns:
        str : the short name
    """
    if not reg_name in regularizers_short_names:
        raise ValueError('the regularizer specified %s is not supported.' % reg_name)
    return regularizers_short_names[reg_name]


def regularization_info(layer_dict):
    # "contractive_regularizer" : ("standard_contractive_regularizer",
    # {"norm": 2, "scale_mean" : 0.1, "scale_covariance" : 0.1})
    reg_info = ""
    contr_reg = layer_dict.get("contractive_regularizer", None)
    if contr_reg is not None:
        reg_info = "r"
        crname, crdict = contr_reg
        if crname == 'contractive_reg_list':
            list_regs = crdict['list_regs']
            for reg_tuple in list_regs:
                reg_info += regularization_info({"contractive_regularizer": reg_tuple})
        else:
            reg_info += get_short_regularization_name(crname)
            if "norm" in crdict:
                reg_info += "_n" + str(crdict["norm"])
            if "scale_mean" in crdict:
                reg_info += "_sm" + str(crdict["scale_mean"])
            if "scale" in crdict:
                reg_info += "_s" + str(crdict["scale"])
            if "scale_covariance" in crdict:
                reg_info += "_sc" + str(crdict["scale_covariance"])

    # TODO add your own regularizer type and extract relevant parameters!

    return reg_info


method_name_short = {

    # activation functions
    'relu':                         'R',
    'elu':                          'E',
    'leaky_relu':                   'LR',
    'LeakyReLU':                   'LR',
    'sigmoid':                      'S',
    'tanh':                         'T',

    # layers
    'dense':                        'D',
    'conv2d':                       'C',
    'max_pooling2d':                'P',
    'flatten':                      '',  # not shown

    # snt modules
    'Linear':                       'D',
    'LinearWN':                     'D',
    'Concatenate':                  'CO',
    'Conv2D':                       'C',
    'Conv2DTranspose':              'CT',
    'Conv2DWN':                     'C',
    'ConvNet2D':                    'CN',
    'ConvNet2DTranspose':           'CNT',
    'ConvDec':                      'CDec',
    'ResEnc':                       'REnc',
    'ResDec':                       'RDec',
    'BatchFlatten':                 '',
    'Identity':                     '',
    'BatchNorm':                    'BN',
    'LayerNorm':                    'LN',
    'BatchReshape':                 'BR',
    'ResUnit':                      'RU',
    'ResNet18':                     'RN18',
    'VGGBlock':                     'V',
    'Sigmoid':                      'S',
    'Tanh':                         'T',
    'MaxPooling2D':                 'P',
    'Dropout':                      'DO',
    'RandomUniform':                'RU',
    'RandomGaussian':               'RG',
    'AveragePooling2D':             'AP',

    # stochastic_models
    'GaussianDiagonal':             'GD',
    'GaussianDiagonalZeroOne':      'GD01',
    'GaussianDiagonalPlusMinusOne': 'GDPM1',
    'Gaussian':                     'G',
    'vonMisesFisher':               'vMF',
    'LogisticDiagonalZeroOne':      'LD01',
    'LogisticDiagonalPlusMinusOne': 'LDPM1',
    'LogitNormalDiagonal':          'LND',
    'LogitNormalDiagonalPlusMinusOne':'LND01', # >>>>>>> TODO this is very confusing, it should be: 01 -> pm1. I don't change it now since some already trained models could not be loaded after
    'Bernoulli':                    'B',
    'BernoulliPlusMinusOne':        'BPM1',
    'OneHotCategorical':            'OHC',

    # initializers
    "glorot_normal_initializer":    'gn',
    "glorot_uniform_initializer":   'gu',
    "xavier_initializer":           'x',
    "truncated_normal_initializer": 't',
    "variance_scaling_initializer": 'v',
    "constant_initializer":         'c',
    "constant":                     'c',
    "random_normal":                'n',

    # custom networks
    "CIFAR10TutorialNetwork":       "CIFAR10TutorialNetwork",

    # regularizers
    "l2_regularizer":               "Ltwo",
    "l1_regularizer":               "Lone",
    "sum_regularizer":              "Sum",
    #keras.regularizers
    "l2":                           "Ltwo",
    "l1":                           "Lone",

    # covariance parameterization
    "softplus":                     'S',
    "linear_softplus":              'LS',
    "exp":                          'E',

    # zero one methods
    "clip":                         'C',
    #"sigmoid":                      's',    #already defined sigmoid "S"

    # clipping gradient
    "clip_by_global_norm":          "GN",
    "clip_by_norm":                 "N",
    "clip_by_value":                "V",
    "none":                         "No",

    # preprocess filtering
    "FromAE":                       "A",
}

#listWithPoints = lambda x: ",".join(re.sub('[( )\[\]]', '', str(list(x))).replace(' ', '').split(","))

def listWithPoints(x):
    if isinstance(x, int):
        x = [x]
    return ",".join(re.sub('[( )\[\]]', '', str(list(x))).replace(' ', '').split(","))

def get_method_id(method_tuple):
    """Creates the id for a method of tensorflow.

    Args:
        method_tuple (tuple): A tuple composed of : (name of the method of tensorflow, kwargs to pass to the method, bool_activation).

    Returns:
        string: the idname of the method that we want to concatenate in the output filenames.

    """
    # ipdb.set_trace()
    # the name could be articulated, since I might want to get the initializers or regularizers
    # from different submodules in tf.
    # e.g. tf.contrib.layers.xavier_initializer
    # the method name I am interested in is the one after the last dot.
    
    if method_tuple is None:
        return "0"

    method_name = method_tuple[0].split('.')[-1]
    method_kwargs = method_tuple[1]

    methodid = method_name_short[method_name]

    if method_name == 'dense':
        methodid += str(method_kwargs['units'])

    elif method_name == 'conv2d':
        methodid += str(method_kwargs['filters']) + 'x' + re.sub('[( )]', '', str(method_kwargs['kernel_size']))

    elif method_name == 'max_pooling2d':
        methodid += re.sub('[( )]', '', str(method_kwargs['pool_size']))

    elif method_name=='AveragePooling2D':
        methodid += re.sub('[( )]', '', str(method_kwargs['pool_size']))

    elif method_name=='AveragePooling2D':
        methodid += re.sub('[( )]', '', str(method_kwargs['pool_size']))

    elif method_name=='flatten':
        pass

    elif method_name == 'Linear':
        #case when it is output layer
        if "output_size" in method_kwargs:
            methodid += str(method_kwargs["output_size"])

    elif method_name == 'Concatenate':
        methodid += str(method_kwargs['node_name'])
        
    elif method_name=='LinearWN':
        # case when it is output layer
        if "output_size" in method_kwargs:
            methodid += str(method_kwargs["output_size"])

        methodid += 'wn'+str(int(method_kwargs['use_weight_norm']))
    
    elif method_name=='Conv2D':
        # case when it is output layer
        if "output_channels" in method_kwargs:
            methodid += str(method_kwargs["output_channels"])

        methodid += 'k'+listWithPoints(method_kwargs['kernel_shape'])

    elif method_name=='Conv2DTranspose':
        # case when it is output layer
        if "output_channels" in method_kwargs:
            methodid += str(method_kwargs["output_channels"])

        methodid += 'o'+listWithPoints(method_kwargs['output_shape'])+'k'+listWithPoints(method_kwargs['kernel_shape'])+'s'+listWithPoints(method_kwargs['stride']) # if you need to changes to 'strides', talk to me (Luigi)

    elif method_name=='Conv2DWN':
        methodid += str(method_kwargs['output_channels'])+'o'+listWithPoints(method_kwargs['kernel_shape'])+\
                        'wn'+str(int(method_kwargs['use_weight_norm']))

    elif method_name=='ResUnit':
        methodid += 'c' + str(method_kwargs['depth'])+'k'+listWithPoints(method_kwargs['kernel_shape'])+\
                        's' + str(method_kwargs['stride'])

    elif method_name == 'VGGBlock':
        methodid += 'c' + str(method_kwargs['channels']) + 'k' + listWithPoints(method_kwargs['kernel_shape']) + \
                    'd' + str(method_kwargs['prob_drop'])
        if method_kwargs.get('logits_size', None) is not None:
            methodid += "l" + listWithPoints(method_kwargs['logits_size'])

    elif method_name=='ResNet18':
        methodid += 'o'+str(method_kwargs['output_size'])+'wn'+str(int(method_kwargs['use_weight_norm']))

    elif method_name=='ConvNet2D':
        methodid += 'o' + listWithPoints(method_kwargs['output_channels']) + 'k' + listWithPoints(
            method_kwargs['kernel_shapes']) + 's' + listWithPoints(method_kwargs['strides'])

    elif method_name=='ConvNet2DTranspose':
        methodid += 'o' + listWithPoints(method_kwargs['output_channels']) + 'k' + listWithPoints(
            method_kwargs['kernel_shapes']) + 's' + listWithPoints(method_kwargs['strides'])

    elif method_name=='ConvDec':
        if method_kwargs.get('linear_first', None) is not None:
            methodid += 'l' + listWithPoints(method_kwargs["linear_first"]["sizes"])
            methodid += 'r' + listWithPoints(method_kwargs["linear_first"]["reshape"])
        methodid += 'c' + listWithPoints(method_kwargs['channels']) \
                    + 'k' + listWithPoints(method_kwargs['kernel_shape']) # + 's' + listWithPoints(method_kwargs['stride'])

    elif method_name in ['ResEnc', 'ResDec']:
        methodid += 'h' + str(method_kwargs['num_hiddens'])
        methodid += 'rl' + str(method_kwargs['num_residual_layers'])
        methodid += 'rh' + str(method_kwargs['num_residual_hiddens'])
        methodid += 'd' + str(method_kwargs['prob_drop'])

    elif method_name == 'BatchFlatten':
        pass

    elif method_name == 'Identity':
        pass

    elif method_name == 'MaxPooling2D':
        methodid += re.sub('[( )]', '', str(method_kwargs['pool_size'])) + 's' + listWithPoints(
            method_kwargs['strides'])


    elif method_name=='Dropout':
        methodid += 'r' + str(method_kwargs['rate']) # tf.layers.dropout

    elif method_name=='Sigmoid':
        pass

    elif method_name=='Tanh':
        pass

    elif method_name == 'RandomUniform':
        methodid += 's' + str(method_kwargs['shape'])
        methodid += 'min' + str(method_kwargs['minval'])
        methodid += 'max' + str(method_kwargs['maxval'])

    # for the moment we don't have mean and covariance as parameters
    elif method_name == 'RandomGaussian':
        methodid += 's' + str(method_kwargs['shape'])

    elif method_name == 'Tanh':
        pass

    elif method_name=='AveragePooling2D':
        methodid += re.sub('[( )]', '', str(method_kwargs['pool_size'])) + 's' + listWithPoints(method_kwargs['strides'])

    #elif method_name=='Dropout':
    #    methodid += 'k'+str(method_kwargs['keep'])

    elif method_name=='Sigmoid':
        raise Exception("why nothing in the name? talk to Luigi")

    elif method_name=='Identity':
        pass

    elif method_name=='BatchReshape':
        pass

    elif method_name=='BatchNorm':
        methodid += '' #+ str(method_kwargs['offset']) + 's' + str(method_kwargs['scale']) + 'd' + str(method_kwargs['decay_rate'])

    elif method_name=='LayerNorm':
        methodid += ''

    elif method_name=='GaussianDiagonal' \
            or method_name=='GaussianDiagonalZeroOne' \
            or method_name=='GaussianDiagonalPlusMinusOne' \
            or method_name=='vonMisesFisher' \
            or method_name=='LogitNormalDiagonal' \
            or method_name=='LogitNormalDiagonalPlusMinusOne' \
            or method_name=='LogisticDiagonalZeroOne' \
            or method_name=='LogisticDiagonalPlusMinusOne':
        # wrapped_module_name, wrapped_module_kwargs = method_kwargs["module_tuple"]
        # LUIGI the lower case is done to increase readibility
        # methodid += "" + method_name_short[wrapped_module_name].lower()
        # if "output_size" in method_kwargs:
        #     methodid += str(method_kwargs["output_size"])
        methodid += "m"+get_method_id(method_kwargs["module_tuple"])

        if "minimal_concentration" in method_kwargs or "minimal_covariance" in method_kwargs: # and method_kwargs["minimal_concentration"] != 1:
            if method_name=='vonMisesFisher':
                methodid += "mc" + str(method_kwargs["minimal_concentration"])
            else:
                methodid += "mc" + str(method_kwargs["minimal_covariance"])
        if "zero_one_method" in method_kwargs and method_kwargs["zero_one_method"] != "sigmoid":
            methodid += "zo" + str(method_name_short[method_kwargs["zero_one_method"]])
        if "scalar_covariance" in method_kwargs:
            scov = method_kwargs["scalar_covariance"]
            if scov == True:
                methodid += "scT"
            elif isinstance(scov, float):
                methodid += "sc" + str(scov)

        if method_name=='LogitNormalDiagonal' or method_name=='LogitNormalDiagonalPlusMinusOne':
            #import pdb;pdb.set_trace()
            if "clip_value" in method_kwargs:
                clip = method_kwargs["clip_value"]
                methodid += "cv" + str(clip)

        #methodid += "_r" + regularization_info(method_kwargs)

    elif method_name=='Bernoulli' or method_name=='BernoulliPlusMinusOne':
        if "output_size" in method_kwargs:
            methodid += str(method_kwargs["output_size"])
        clip = method_kwargs["clip_value"]
        methodid += "cv" + str(clip)

    elif method_name=='OneHotCategorical':
        if "output_size" in method_kwargs:
            methodid += str(method_kwargs["output_size"])
        clip = method_kwargs["clip_value"]
        methodid += "cv" + str(clip)

    elif method_name == 'CIFAR10TutorialNetwork':
         pass

    elif "variance_scaling_initializer" in method_name:
        pass

    elif "glorot_normal_initializer" in method_name or "glorot_uniform_initializer" in method_name:
        pass

    elif "truncated_normal_initializer" in method_name or "random_normal" in method_name:
        methodid += str(method_kwargs['stddev'])

    elif "constant_initializer" in method_name or "constant" in method_name:
        methodid += str(method_kwargs['value'])

    elif method_name in ["l1_regularizer", "l2_regularizer", "sum_regularizer"]:
        methodid += str(method_kwargs["scale"])

    elif method_name in ["l1", "l2"]:
        methodid += str(method_kwargs["l"])

    elif method_name == "softplus":
        pass

    elif method_name == "linear_softplus":
        pass

    elif method_name == "exp":
        pass

    # PREPROCESSING SECTION used to prefilter transform an image with some method
    elif method_name == "FromAE":
        methodid += hash_this(method_kwargs["filename"], trunc=3)
        methodid += "t" + str(method_kwargs["transform_prob"])
        methodid += "n" + str(method_kwargs["noisy_transform_prob"])

    # Here implement your favourite method
    # elif      :
    #

    else:
        print('----------------------')
        print('ERROR ', method_name)
        raise ValueError("id rule for `%s` has to be implemented." % method_name)

    # support for contractive only in some layers for the moment, but it could be easily extended,
    # just add your layer and test it
    if "contractive_regularizer" in method_kwargs:
        if method_name == "Linear" or method_name == "GaussianDiagonal" or method_name == "GaussianDiagonalPlusMinusOne":
            methodid += regularization_info(method_kwargs)
        else:
            raise ValueError("contractive_regularizers not supported for `%s`." % method_name)

    return methodid

def get_clipping_id(clipping_tuple):
    method = clipping_tuple[0]
    if not method:
        return method_name_short["no"]
    else:
        value = clipping_tuple[1]["value"]
        return method_name_short[method] + "{:.4g}".format(value)

def eval_method_from_tuple(module, method_tuple, instantiate=True):
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

#     import pdb;pdb.set_trace()
    if instantiate:
        return method_fn(**method_tuple[1])
    else:
        return method_fn


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


def try_load_class_from_modules(class_path, modules):
    LayerClass = None

    for module in modules:
        try:
            LayerClass = load_method_fn_from_method_path(module, class_path)
        except:
            pass

    if LayerClass is None:
        raise Exception("problem loading class: {:}, not found in modules {:}".format(class_path, modules))

    return LayerClass


def load_class(module_plus_class, relative=False, base_path=''):
    # assemble class path
    class_path = ''
    # if class_base_path prepend this to the path
    if base_path:
        class_path = base_path
        # if the prepended path does not finish with a dot add it
        if class_path[-1] != '.':
            class_path += '.'
    class_path += module_plus_class

    # split in all modules and class
    modulesname, classname = class_path.rsplit('.', 1)
    if relative:
        modulesname = class_path.split('.', 1)[1]

    my_module = importlib.import_module(modulesname)
    my_class = getattr(my_module, classname)
    return my_class

def load_module(module, relative=False, base_path=''):
    # assemble class path
    class_path = ''
    #if class_base_path prepend this to the path
    if base_path:
        class_path = base_path
        #if the prepended path does not finish with a dot add it
        if class_path[-1] != '.':
            class_path += '.'
    class_path += module

    modulesname = class_path
    if relative:
        modulesname = class_path.split('.', 1)[1]

    my_module = importlib.import_module(modulesname)
    return my_module

def eval_file(file_name_path):
    with open(file_name_path, 'r') as fstream:
        return eval(fstream.read())

def freeze_graph_create_pb(session,
                           output_names=None,
                           variable_names_whitelist=None,
                           variable_names_blacklist=None,
                           output_filename = None,
                           clear_devices=True):

        """
        Freezes the state of a session into a pruned computation graph.

        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param session The TensorFlow session to be frozen.
        @param variable_names_whitelist A list of variable to be frozen (default all)
        @param variable_names_blacklist A list of variable names that should not be frozen,
                                        or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        graph = session.graph
        with graph.as_default():
            if variable_names_whitelist is not None:
                freeze_var_names = variable_names_whitelist
            else:
                freeze_var_names = [v.op.name for v in tf.global_variables()]
            freeze_var_names = list(set(freeze_var_names).difference(variable_names_blacklist or []))

            if output_names is None:
                output_names = [v.op.name for v in tf.global_variables()]

            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""

            frozen_graph = tf.graph_util.convert_variables_to_constants(session,
                                                                        input_graph_def,
                                                                        output_names,
                                                                        freeze_var_names)

        # finally we serialize and dump the output graph to the filesystem
        if output_filename is not None:
            with tf.gfile.GFile(output_filename, "wb") as f:
                f.write(frozen_graph.SerializeToString())

        print("freezing and creating pb file: %d ops in the final graph." % len(frozen_graph.node))
        #return frozen_graph


def unpack_dict_of_lists(dictionary):
    return [dict(zip(dictionary.keys(), p)) for p in product(*map(make_list, dictionary.values()))]


def apply_resize(x, intermediate_size):
    intermediate_x = tf.image.resize(x, tf.constant([intermediate_size, intermediate_size]))
    resized_x = tf.image.resize(intermediate_x, tf.shape(x)[1:3])
    return resized_x


def tf_f1_score(y_true, y_pred):
    """Computes 3 different f1 scores, micro macro
    weighted.
    micro: f1 score accross the classes, as 1
    macro: mean of f1 scores per class
    weighted: weighted average of f1 scores per class,
            weighted from the support of each class

    from https://stackoverflow.com/questions/45287169/tensorflow-precision-recall-f1-multi-label-classification

    Args:
        y_true (Tensor): labels, with shape (batch, num_classes)
        y_pred (Tensor): model's predictions, same shape as y_true

    Returns:
        tuple(Tensor): (micro, macro, weighted)
                    tuple of the computed f1 scores
    """

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1.) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2. * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted
