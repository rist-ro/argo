import tensorflow as tf
from tensorflow_probability import distributions as tfd

import types

import pdb

from .ELBO import ELBO

def logsumexp(x):
    x_max = tf.reduce_max(x, [1], keepdims=True)
    return tf.reshape(x_max, [-1]) + tf.log(tf.reduce_sum(tf.exp(x - x_max), [1]))

# def compute_lowerbound(log_pxz, sum_kl_costs, k=1):
#     if k == 1:
#         return sum_kl_costs - log_pxz
#
#     # the following is a just a IS estimation, which is not the ELBO cost function
#
#     # log 1/k \sum p(x | z) * p(z) / q(z | x) = -log(k) + logsumexp(log p(x|z) + log p(z) - log q(z|x))
#     log_pxz = tf.reshape(log_pxz, [-1, k])
#     sum_kl_costs = tf.reshape(sum_kl_costs, [-1, k])
#     return - (- tf.log(float(k)) + logsumexp(log_pxz - sum_kl_costs))
#
# def discretized_logistic(mean, logscale, binsize=1/256.0, sample=None):
#     scale = tf.exp(logscale)
#     sample = (tf.floor(sample / binsize) * binsize - mean) / scale
#     logp = tf.log(tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
#     return tf.reduce_sum(logp, [1, 2, 3])

NUMTOL = 1e-7
def integral_log_prob(distribution, sample, binsize=1/256.0):
    xi = tf.floor(sample / binsize) * binsize
    xip1 = xi + binsize
    logp = tf.log(distribution.cdf(xip1) - distribution.cdf(xi) + NUMTOL)
    return tf.reduce_sum(logp, [1, 2, 3])


class ELBO_IAF(ELBO):

    def __init__(self, beta=1.0, warm_up_method=None, kl_min=0.25, name="ELBO_IAF"):
        super().__init__(beta=beta, warm_up_method=warm_up_method, name = name)
        self._kl_min = kl_min
        # self.dec_log_stdv = tf.get_variable("dec_log_stdv", initializer=tf.constant(0.0))
    
    def create_id(self, cost_fuction_kwargs):

        # cost_fuction_kwargs[0] is the cost function name
        _id = super().create_id(cost_fuction_kwargs)
        _id += "_km"+str(cost_fuction_kwargs["kl_min"])
        
        return _id
    
    def compute_kl_of_layer(self, x_repeated, prior, posterior, z1, flow_chain_params, zN):
        
        # logqs = posterior.logps(z1)
        logqs = posterior.log_prob(z1)
        for flow_params in flow_chain_params:
            logqs += flow_params['lnsd']
        
        # logps = prior.logps(zN)
        logps = prior.log_prob(zN)
        kl_cost = logqs - logps
        
        if self._kl_min > 0:
            # [0, 1, 2, 3] -> [0, 1] -> [1] / (b * k)
            kl_ave = tf.reduce_mean(tf.reduce_sum(kl_cost, [1, 2]), [0], keepdims=True)
            kl_ave = tf.maximum(kl_ave, self._kl_min)
            kl_ave = tf.tile(kl_ave, [tf.shape(x_repeated)[0], 1]) #self._batch_size
            kl_obj = tf.reduce_sum(kl_ave, [1])
        else:
            kl_obj = tf.reduce_sum(kl_cost, [1, 2, 3])
        
        kl_cost = tf.reduce_sum(kl_cost, [1, 2, 3])
        
        return kl_obj, kl_cost

    def _build(self, model):
        
        kl_cost = 0.0
        kl_obj = 0.0
        
        x = model.x
        x_repeated = model.x_repeated
        model_visible = model._model_visible
        for prior, posterior, z1, flow_chain_params, zN in zip(model.priors,
                                                         model.posteriors,
                                                         model.z1s,
                                                         model.flows_params,
                                                         model.zNs):
            
            layer_obj, layer_cost = self.compute_kl_of_layer(x_repeated,
                                                         prior,
                                                         posterior,
                                                         z1,
                                                         flow_chain_params,
                                                         zN)
            kl_obj += layer_obj
            kl_cost += layer_cost
        
        
        # log_pxz = discretized_logistic(x_reconstruction_node, self.dec_log_stdv, sample=x_repeated)
        log_pxz = integral_log_prob(model_visible, x_repeated)
        
        # NB originally it was reduce_sum
        # TODO ?
        #obj = tf.reduce_mean(kl_obj - log_pxz)

        
        # beta = tf.placeholder_with_default(self._beta, shape=(), name='beta_regularizer')
        # warm_up = self.get_warm_up_coefficient(self._warm_up_method, vae)
        #
        # # The loss is composed of two terms:
        # #
        # # 1.) The reconstruction loss (the negative log probability
        # #     of the input under the reconstructed distribution
        # #     induced by the decoder in the data space).
        # #     This can be interpreted as the number of "nats" required
        # #     for reconstructing the input when the activation in latent
        # #     is given.
        #
        # reconstruction_loss = self.reconstruction_loss(x_target, n_z_samples, model_visible)
        #
        # # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        # #     between the distribution in latent space induced by the encoder on
        # #     the data and some prior. This acts as a kind of regularizer.
        # #     This can be interpreted as the number of "nats" required
        # #     for transmitting the the latent space distribution given
        # #     the prior.
        #
        # latent_loss = beta * self.latent_loss(gaussian_model_latent, prior)
        #
        # cost = warm_up * latent_loss + reconstruction_loss

        # NB originally is was tf.add_n
        _k = model._network._k

        reconstruction_loss = tf.reduce_mean(log_pxz)
        latent_loss_obj = tf.reduce_mean(kl_obj)
        latent_loss_cost = tf.reduce_mean(kl_cost)

        warm_up = self.get_warm_up_coefficient(self._warm_up_method, model)

        loss =  warm_up * self.beta * latent_loss_obj - reconstruction_loss
        #tf.reduce_mean(compute_lowerbound(log_pxz, kl_cost, _k))
        
        return loss, [[reconstruction_loss], [latent_loss_obj, latent_loss_cost]], [["RL"], ["max(KL,kl_min)", "KL"]], [{"fileName" : "cost_function_RL"}, {"fileName" : "cost_function_KL"}]
    
