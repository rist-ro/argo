import tensorflow as tf
import tensorflow_probability as tfp
import re
from ..utils.argo_utils import eval_method_from_tuple, try_load_class_from_modules, method_name_short
from ..regularizers.load_regularizer import load_regularizer
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from functools import partial
from .FromCallable import FromCallable

MINIMAL_COVARIANCE = 1e-4


# def get_activation(opts):
#     activation = eval_method_from_tuple(tf.nn, opts["activation"])
#     return activation

def get_keras_activation(activation_tuple):
    activation_name, activation_kwargs = activation_tuple

    activation_func = getattr(tf.keras.activations, activation_name)
    ActivationLayer = FromCallable(activation_func, activation_kwargs)

    return ActivationLayer

def act_id(activation_tuple):
    activation_name, activation_kwargs = activation_tuple

    act_dict = {
        'relu' : 'R'
     }

    try:
        _id = act_dict[activation_name]

        if activation_name=='relu':
            if activation_kwargs.get('alpha', None) is not None:
                _id += "a{:.1f}".format(activation_kwargs['alpha'])
    except:
        raise Exception("Activation not supported in name id, found {:} - {:}".format(activation_name, activation_kwargs))

    return _id

    # kernel_untr_scale_initializer_mean = -9.,
    # kernel_untr_scale_initializer_stddev = 1e-2,
    # kernel_loc_initializer_mean = 0.,
    # kernel_loc_initializer_stddev = 1e-2,
    # kernel_lambda_l2_loc = 1e-5,
    # kernel_lambda_l2_untr_scale = 1e-5,
    # kernel_posterior_logscale_constraint_min = -1000.,
    # kernel_posterior_logscale_constraint_max = tf.math.log(0.2),
    # bias_loc_initializer_mean = 1.,
    # bias_lambda_l2_loc = 0.


def parse_init_reg_kwargs_bayes(kwargs):
    if kwargs is None:
        return None

    posterior_kwargs = kwargs["posterior"]

    activity_regularizer = eval_method_from_tuple(tf, posterior_kwargs["activity_regularizer"])

    kernel_loc_initializer = eval_method_from_tuple(tf, posterior_kwargs["kernel_loc_initializer"])
    kernel_untr_scale_initializer = eval_method_from_tuple(tf, posterior_kwargs["kernel_untr_scale_initializer"])

    kernel_loc_regularizer = eval_method_from_tuple(tf, posterior_kwargs["kernel_loc_regularizer"])
    kernel_untr_scale_regularizer = load_regularizer(posterior_kwargs["kernel_untr_scale_regularizer"])

    bias_loc_initializer = eval_method_from_tuple(tf, posterior_kwargs["bias_loc_initializer"])
    bias_loc_regularizer = eval_method_from_tuple(tf, posterior_kwargs["bias_loc_regularizer"])

    kernel_untr_scale_constraint_min = -1000
    kernel_untr_scale_constraint_max = float(posterior_kwargs["kernel_untr_scale_constraint_max"])

    def kernel_untr_scale_constraint(tensor):
        return tf.clip_by_value(tensor,
                                kernel_untr_scale_constraint_min,
                                kernel_untr_scale_constraint_max)


    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
                    loc_initializer = kernel_loc_initializer,
                    untransformed_scale_initializer = kernel_untr_scale_initializer,
                    untransformed_scale_constraint = kernel_untr_scale_constraint,
                    loc_regularizer = kernel_loc_regularizer,
                    untransformed_scale_regularizer = kernel_untr_scale_regularizer
    )

    bias_posterior_fn = tfp.layers.default_mean_field_normal_fn(
                    loc_initializer = bias_loc_initializer,
                    loc_regularizer = bias_loc_regularizer,
                    is_singular = True
    )

    prior_kwargs = kwargs["prior"]

    kernel_prior_fn = load_kernel_prior_fn(prior_kwargs)


    kwargs_bayes = {
        "kernel_prior_fn": kernel_prior_fn,
        "kernel_posterior_fn": kernel_posterior_fn,
        "bias_posterior_fn": bias_posterior_fn,
        "activity_regularizer": activity_regularizer,
    }

    return kwargs_bayes

def load_kernel_prior_fn(prior_kwargs):

    default_prior = prior_kwargs["default"]

    if default_prior:
        if len(prior_kwargs)>1:
            raise ValueError("If prior is default, no other options should be set in the prior kwargs")

        kernel_prior_fn = tfp.layers.default_multivariate_normal_fn

    else:

        trainable = prior_kwargs["trainable"]

        if trainable:
            prior_kernel_loc_initializer = eval_method_from_tuple(tf, prior_kwargs["kernel_loc_initializer"])
            prior_kernel_untr_scale_initializer = eval_method_from_tuple(tf, prior_kwargs["kernel_untr_scale_initializer"])
            kernel_prior_fn = tfp.layers.default_mean_field_normal_fn(
                            loc_initializer = prior_kernel_loc_initializer,
                            untransformed_scale_initializer = prior_kernel_untr_scale_initializer
            )

        # constants do not seems to work, no idea why!
        # elif prior_kwargs["kernel_loc_initializer"][0]=="initializers.constant" and \
        #         prior_kwargs["kernel_untr_scale_initializer"][0]=="initializers.constant":
        #
        #     prior_kernel_loc_value = prior_kwargs["kernel_loc_initializer"][1]["value"]
        #     prior_kernel_untr_scale_value = prior_kwargs["kernel_untr_scale_initializer"][1]["value"]
        #
        #     def create_fixed_gaussian_prior(dtype, shape, name, trainable, add_variable_fn):
        #
        #         prior_loc = tf.constant(prior_kernel_loc_value, dtype=dtype,
        #                                 shape=shape, name='kernel_prior_loc')
        #
        #         # prior_untr_scale = tf.constant(prior_kernel_untr_scale_value, dtype=dtype,
        #         #                                name='kernel_prior_untransformed_scale')
        #
        #         dist = normal_lib.Normal(
        #             loc=prior_loc, scale=tf.nn.softplus(dtype.as_numpy_dtype(prior_kernel_untr_scale_value)))
        #
        #         batch_ndims = tf.size(input=dist.batch_shape_tensor())
        #
        #         return independent_lib.Independent(
        #             dist, reinterpreted_batch_ndims=batch_ndims)
        #
        #     kernel_prior_fn = create_fixed_gaussian_prior

        else:
            prior_kernel_loc_initializer = eval_method_from_tuple(tf, prior_kwargs["kernel_loc_initializer"])
            prior_kernel_untr_scale_initializer = eval_method_from_tuple(tf, prior_kwargs["kernel_untr_scale_initializer"])

            def create_fixed_gaussian_prior(dtype, shape, name, trainable, add_variable_fn):

                prior_loc = add_variable_fn('kernel_prior_loc', shape=shape, dtype=dtype, trainable=False,
                                                      initializer=prior_kernel_loc_initializer)
                prior_untr_scale = add_variable_fn('kernel_prior_untransformed_scale', shape=(), dtype=dtype,
                                                             trainable=False,
                                                             initializer=prior_kernel_untr_scale_initializer)

                dist = normal_lib.Normal(
                    loc=prior_loc, scale=tf.nn.softplus(prior_untr_scale))

                batch_ndims = tf.size(input=dist.batch_shape_tensor())

                return independent_lib.Independent(
                    dist, reinterpreted_batch_ndims=batch_ndims)
                # distr = tfp.distributions.MultivariateNormalDiag(loc=prior_loc, scale_diag=tf.nn.softplus(prior_untr_scale), name=name)
                # return distr

            kernel_prior_fn = create_fixed_gaussian_prior

    return kernel_prior_fn

def parse_init_reg_kwargs(kwargs):
    if kwargs is None:
        return None

    #initializers
    kernel_initializer = eval_method_from_tuple(tf, kwargs["kernel_initializer"])
    bias_initializer = eval_method_from_tuple(tf, kwargs["bias_initializer"])

    # regularizers
    kernel_regularizer = eval_method_from_tuple(tf, kwargs["kernel_regularizer"])
    bias_regularizer = eval_method_from_tuple(tf, kwargs["bias_regularizer"])
    activity_regularizer = eval_method_from_tuple(tf, kwargs["activity_regularizer"])

    kwargs = {
        "kernel_initializer" : kernel_initializer,
        "bias_initializer": bias_initializer,

        "kernel_regularizer": kernel_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
    }

    return kwargs


name_short = {
    'BayesianVgg':             'BVGG',
    # 'bayesian_vgg':            'BVGG',
    # 'bayesian3D_vgg':            'BVGG3D',
    # 'bayesian_vgg_LSS':            'BVGGLSS',
    # 'bayesianTrick_vgg':       'BVGGT',
    # 'bayesian_resnet':         'BResNet',
    'BayesianResNet':          'BResNet',
    'BayesianInception':        'BIncep',
    'BayesianInception_Basic':  'BIncepB',
    'BayesianInception_Basic_residual':  'BIncepBres',
    'MultivariateNormalDiag':  'D',
    'MultivariateNormalTriL':  'TriL',
    'OneHotCategorical':       'Cat',
    'bayesian_alex_net':       'Alex',
    'BiLSTM':                  'BiLSTM',
    'name_net':                ''
}


def get_network_id(method_tuple):
    """Creates the id for a keras network.

    Args:
        method_tuple (tuple): A tuple composed of : (name of the keras model builder function, kwargs to pass to the function).

    Returns:
        string: the idname of the keras network that we want to concatenate in the output filenames.

    """

    listWithPoints = lambda x: ".".join(re.sub('[( )\[\]]', '', str(x)).replace(' ', '').split(","))

    method_name = method_tuple[0]
    method_kwargs = method_tuple[1]

    methodid = name_short[method_name]

    if method_name in ['BayesianVgg', 'BayesianInception','BayesianInception_Basic','BayesianInception_Basic_residual']:
        methodid += "_fl" + str(int(method_kwargs['flipout']))
        methodid += "_rn" + str(int(method_kwargs['renorm']))
        if hasattr(method_kwargs, 'logits_size'):
            methodid += "_l" + str(method_kwargs['logits_size'])

        #methodid += "_f"+listWithPoints(method_kwargs['filters'])
        methodid += "_"+str(method_kwargs['pooling']).lower()[0]+"p"

    elif method_name == 'bayesianTrick_vgg':
        if hasattr(method_kwargs, 'logits_size'):
            methodid += "_l"+str(method_kwargs['logits_size'])

    elif method_name in ['bayesian_resnet', 'BayesianResNet']:
        methodid += "_fl" + str(int(method_kwargs['flipout']))
        methodid += "_rn" + str(int(method_kwargs['renorm']))
        if hasattr(method_kwargs, 'logits_size'):
            methodid += "_l" + str(method_kwargs['logits_size'])

        methodid += "_f"+listWithPoints(method_kwargs['filters'])
        activation_name = method_kwargs['activation'][0]
        methodid += "_a" + method_name_short[activation_name]

    elif method_name == 'name_net':
        methodid += str(method_kwargs['name'])

    elif method_name == 'MultivariateNormalDiag' or method_name == 'MultivariateNormalTriL':
        methodid += "_b"+str(int(method_kwargs['bayesian_layers']))
        methodid += "_fl" + str(int(method_kwargs['flipout']))

    elif method_name == 'OneHotCategorical':
        pass

    elif method_name == 'BiLSTM':
        pass

    else:
        print('----------------------')
        print('ERROR ', method_name)
        raise ValueError("id rule for keras network `%s` has to be implemented." % method_name)

    return methodid

def get_renorm_clipping(dtype = tf.float32):
    # from batch renormalization paper
    global_step = tf.train.get_or_create_global_step()

    with tf.name_scope("renorm_clipping"):

        init_decay_step = 5000
        decay_global_step = global_step - init_decay_step

        rmax_initial = tf.constant(1, dtype=dtype)
        rmax_final = tf.constant(3, dtype=dtype)
        rmax_relax = tf.train.polynomial_decay(rmax_initial,
                                               decay_global_step,
                                               40000 - init_decay_step,
                                               end_learning_rate=rmax_final,
                                               name="rmax")

        dmax_initial = tf.constant(0, dtype=dtype)
        dmax_final = tf.constant(5, dtype=dtype)
        dmax_relax = tf.train.polynomial_decay(dmax_initial,
                                               decay_global_step,
                                               25000 - init_decay_step,
                                               end_learning_rate=dmax_final,
                                               name="dmax")

        rmax = tf.cond(global_step < init_decay_step, lambda: rmax_initial, lambda: rmax_relax)
        dmax = tf.cond(global_step < init_decay_step, lambda: dmax_initial, lambda: dmax_relax)

        renorm_clipping = {
            "rmax": rmax,
            "rmin": 1 / rmax,
            "dmax": dmax
        }

    return renorm_clipping

def parse_layer(layer_tuple):
    # TODO how to implement decorators
    decorators = []
    try:
        layer_name, layer_kwargs, decorators = layer_tuple
    except ValueError as e:
        layer_name, layer_kwargs = layer_tuple

    modules = [tf, tfp]

    # load base class and instantiate
    BaseClass = try_load_class_from_modules(layer_name, modules)
    layer = BaseClass(**layer_kwargs)

    # for each decorator decorate it
    for decorator_name, decorator_kwargs in decorators:
        DecoratorClass = try_load_class_from_modules(decorator_name, modules)
        layer = DecoratorClass(layer=layer, **decorator_kwargs)

    return layer

