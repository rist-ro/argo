import tensorflow_probability as tfp
tfb = tfp.bijectors
import tensorflow as tf

import importlib
import numpy as np


def init_once(x, name):
    with tf.variable_scope('build_flow', reuse=tf.AUTO_REUSE):
        return tf.get_variable(name, initializer=x, trainable=False, dtype=tf.int32)


def build_flow(flow_params, flow_size): #, module_path = ""):
    flow_size = int(flow_size) # in case it is a tf Dimension
    flow_name = flow_params['name']
    num_bijectors = flow_params['num_bijectors']
    hc = flow_params['hidden_channels']
    permute = flow_params.get('permute', True)
    # output_size = flow_params['output_size']

    bijectors = []
    flow_vars = []

    for i in range(num_bijectors):
        bij = None
        if flow_name == 'NVP':
            #         bijectors.append(NVPCoupling(D=2, d=1, layer_id=i))
            bij = tfb.RealNVP(
                            num_masked=int(0.5*flow_size),
                            shift_and_log_scale_fn=tfb.real_nvp_default_template(hidden_layers=[hc, hc]))
        elif flow_name == 'MAF':
            # shift_and_log_scale = tfb.masked_autoregressive_default_template(hidden_layers=[hc, hc])
            bij = tfb.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(hidden_layers=[hc, hc]), name='maf_bijector-{}'.format(i))
        elif flow_name == 'IAF':
            bij = tfb.Invert(tfb.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                            hidden_layers=[hc, hc]), name='iaf_bijector-{}'.format(i)))
        else:
            raise Exception("flow_name: `{:}` not recognized".format(flow_name))

        bijectors.append(bij)

        if permute:
            perm = tfb.Permute(permutation=init_once(np.random.permutation(flow_size).astype('int32'),
                                                     name='flow_permutation_{}'.format(i)))
            bijectors.append(perm)

    # Discard the last Permute layer.
    if permute:
        bijectors = bijectors[:-1]

    flow_bijector = tfb.Chain(list(reversed(bijectors)))

    return flow_bijector


    # dist = tfp.distributions.TransformedDistribution(
    #                             distribution=base_dist,
    #                             bijector=flow_bijector)

