import importlib

import tensorflow as tf

from argo.core.utils.argo_utils import eval_method_from_tuple, CUSTOM_REGULARIZATION

def replicate(param, n_z_samples):
    num_shapes = len(param.shape.as_list()[1:])
    ones = [1] * num_shapes
    param_replicate = tf.tile(param, [n_z_samples] + ones)
    return param_replicate

def autoencoding_regularizer(model, distance_function, scale):
    """
        Custom regularizer which minimizes dissimilarity(enc(x), enc(rec(x)) given by distance_function
        where distance_function \in {euclidean, wasserstein, kl, fisher}
    """

    try:
        module_path = 'argo.core.utils.distances'
        reg_module = importlib.import_module(module_path)
        dissimilarity = eval_method_from_tuple(reg_module, (distance_function,), instantiate=False)
    except ImportError as e:
        raise ImportError("regularizer %s not found" % distance_function) from e

    if model.mask is not None:
        mask = replicate(model.mask, model.n_z_samples)
        reconstruction_node = (mask * (model.x_reconstruction_node + 1) / 2) * 2 - 1
    else:
        reconstruction_node = model.x_reconstruction_node

    encoding_of_reconstruction_params = model._network.encoder_module(reconstruction_node).params()
    # replicate in case of multiple z samples
    approximate_posterior_params = [replicate(p, model.n_z_samples)
                                    for p in model._approximate_posterior_params]

    # in case of the Euclidean distance, we compute the distance between mean vectors
    if distance_function == 'euclidean':
        approximate_posterior_params = approximate_posterior_params[0]
        encoding_of_reconstruction_params = encoding_of_reconstruction_params[0]

    reg_node = dissimilarity(approximate_posterior_params,
                             encoding_of_reconstruction_params)

    regularizer = tf.reduce_mean(reg_node)
    scaled_regularizer = tf.multiply(scale, regularizer, name=distance_function + "_custom_reg")

    tf.add_to_collection(CUSTOM_REGULARIZATION, scaled_regularizer)

    return scaled_regularizer, [regularizer], "autoencoding_regularizer"
