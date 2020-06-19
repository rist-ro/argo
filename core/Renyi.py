import tensorflow as tf
from tensorflow_probability import distributions as tfd

import types

import pdb

from argo.core.Network import AbstractModule

# moved from VAE
# TODO-ARGO2 all these default values should not be hard coded but handled in the default_params
# self.warm_up_method = opts.get("warm_up_method")
# be careful with initialization in the following, since values have to be float

# TODO-ARGO2 this parameter handling is very confusing.. maybe a class for cost function should handle its own default parameters,
# TODO-ARGO2 why do I need to handle them at this level?
# alpha parameter for *renyi

'''
if opts["cost_function"]["cost"]=="renyi" or opts["cost_function"]["cost"]=="irenyi":
    if "alpha" not in opts["cost_function"]:
        opts["cost_function"]["alpha"] = 1.0
    self.alpha_default = opts["cost_function"]["alpha"]

# beta parameter for *elbo
if opts["cost_function"]["cost"]=="elbo" or opts["cost_function"]["cost"]=="ielbo":
    if "beta" not in opts["cost_function"]:
        opts["cost_function"]["beta"] = 1.0
    self.beta_default = opts["cost_function"]["beta"]

# h parameter for ielbo and irenyi
if opts["cost_function"]["cost"]=="ielbo" or opts["cost_function"]["cost"]=="irenyi":
    if "h" not in opts["cost_function"]:
        opts["cost_function"]["h"] = 0.01
    self.h_default = opts["cost_function"]["h"]

# normalize for ielbo and irenyi
if opts["cost_function"]["cost"]=="ielbo" or opts["cost_function"]["cost"]=="irenyi":
    if "normalize" not in opts["cost_function"]:
        opts["cost_function"]["normalize"] = 0
    self.h_normalize = opts["cost_function"]["normalize"]

'''


class Renyi(AbstractModule):

    def __init__(self, beta=1.0, warm_up_method=None, name="ELBO"):
        super().__init__(name = name)

        #self._warm_up_method = warm_up_method
        #self.beta = tf.placeholder_with_default(float(beta), shape=(), name='beta_regularizer')
        self.alpha = tf.placeholder_with_default(float(alpha), shape=(), name='alpha_renyi')
    
    def create_id(self, cost_fuction_kwargs):
        
        # cost_fuction_kwargs[0] is the cost function name

        _id = "Renyi_a" + str(cost_fuction_kwargs["alpha"])
        #_id += self.create_warm_up_id(cost_fuction_kwargs)

        return _id

    '''
    def create_warm_up_id(self, cost_fuction_kwargs):
        
        if "warm_up_method" not in cost_fuction_kwargs or cost_fuction_kwargs["warm_up_method"] is None:
            warm_up = "0"
        elif cost_fuction_kwargs["warm_up_method"][0] == "warm_up":
            warm_up = "W" + str(cost_fuction_kwargs["warm_up_method"][1]["epoch"])
        elif cost_fuction_kwargs["warm_up_method"][0] == "inverse_warm_up":
            warm_up = "IW" + str(cost_fuction_kwargs["warm_up_method"][1]["epoch"])
        else:
            raise Exception("warm_up_method not recognized")
        _id = '_wu' + warm_up
        
        return _id
    
    def get_warm_up_coefficient(self, warm_up_method, vae_model):
        # Warm up: gradually introduce the KL divergence in the cost function,
        # as suggested in Ladder VAE paper, and also others
        #
        # warm_up_method  None             = no warm up
        #                "inverse_warm_up" = inverse warm up (scale of KL decreases until it becomes 1)
        #                "warm_up"         = warm up (scale of KL increases until it becomes 1)

        # initialize the warm_up coefficient to be placed in front of the KL term for ELBO and IELBO
        if warm_up_method is None:
            warm_up = tf.constant(1.)
                
        elif warm_up_method[0] == "warm_up":
            global_step = vae_model.global_step
            warm_up_final_step = warm_up_method[1]["epoch"] * vae_model.n_batches_per_epoch
            # increase the KL scale from 0 to 1, over self.warm_up_method[1]["epoch"] epochs
            warm_up = tf.cond(global_step > warm_up_final_step,
                                lambda: tf.constant(1.),
                                lambda: tf.cast(global_step - 1, dtype=tf.float32) /
                                                    tf.constant(warm_up_final_step-1, dtype=tf.float32)
                              )
        
        elif warm_up_method[0] == "inverse_warm_up":
            
            # TODO-LUIGI what is below needs to be reimplemented
            raise Exception("what is below needs to be reimplemented")
            warm_up_final_step = warm_up_method[1]["epoch"] * vae_model.n_batches_per_epoch
                
            # decrease the KL scale from 10 (epoch 0) to 1 (epoch warm_up_method[1]["epoch"])
            # in general the formula is n - ((n-1)*epoch) / warm_up_method[1]["epoch"]
            warm_up = tf.constant(10, dtype=tf.float32) - tf.cast(9*global_epoch, dtype=tf.float32) / tf.constant(float(warm_up_method[1]["epoch"]), dtype=tf.float32)

        else:
            raise Exception("warm_up_method not recognized")

        return warm_up
    '''
    
    def _build(self, vae):
        prior =  vae._prior
        gaussian_model_latent = vae._gaussian_model_latent
        model_visible = vae._model_visible
        x_target =  vae.x_target
        n_z_samples = vae.n_z_samples

        #warm_up = self.get_warm_up_coefficient(self._warm_up_method, vae)

        #reconstruction_loss = self.reconstruction_loss(x_target, n_z_samples, model_visible)
        #latent_loss = self.beta * self.latent_loss(gaussian_model_latent, prior)
        renyi_loss = reconstruction_loss(x_target, n_z_samples, model_visible, gaussian_model_latent, prior) # SEPTIMIA

        # Check signs SEPTIMIA
        cost = -renyi_loss

        # Check signs SEPTIMIA
        return cost, [renyi_loss], ["RenyiRL"]


    def renyi_loss(self, x_target, n_z_samples, model_visible, gaussian_model_latent, prior):
        pass
        
    '''
    def reconstruction_loss(self, x_target, n_z_samples, model_visible):
        
        # with tf.variable_scope('ELBO/reconstruction_loss'):
        # no need for ELBO, sonnet module is already adding that, the line above would produce:
        # ELBO/ELBO/reconstruction_loss/node_created
        with tf.variable_scope('reconstruction_loss'):

            # 1) the log_pdf is computed with respect to distribution of the visible
            #    variables obtained from the target of input of the graph (self.x_target)

            # can I avoid replicate? maybe not..
            input_shape = x_target.shape.as_list()[1:]
            ones = [1] * len(input_shape)
            x_replicate = tf.tile(x_target, [n_z_samples] + ones)

            reconstr_loss = - model_visible.log_prob(x_replicate)
            
            # #before
            # reconstr_loss = tf.reshape(reconstr_loss, [n_z_samples, -1]+input_shape)
            # all_axis_but_first_2 = list(range(len(reconstr_loss.shape)))[2:]
            # #independent p for each input pixel
            # log_p = tf.reduce_sum(reconstr_loss, axis=all_axis_but_first_2)
            # #average over the samples
            # mean_reconstr_loss = tf.reduce_mean(log_p, axis=0)
            
            #now (ready for arbitrary intermediate samplings)
            all_axis_but_first = list(range(len(reconstr_loss.shape)))[1:]
            #independent p for each input pixel
            log_p = tf.reduce_sum(reconstr_loss, axis=all_axis_but_first)
            #average over all the samples and the batch (they are both stacked on the axis 0)
            mean_reconstr_loss = tf.reduce_mean(log_p, axis=0, name="reconstruction_loss")
            
            # self.log_p_x_z = mean_reconstr_loss

        return mean_reconstr_loss
    
    def latent_loss(self, gaussian_model_latent, prior):
   
        # with tf.variable_scope('ELBO/reconstruction_loss'):
        # no need for ELBO, sonnet module is already adding that, the line above would produce:
        # ELBO/ELBO/reconstruction_loss/node_created
        with tf.variable_scope('latent_loss'):
            # 1) cast is required
            # 2) the KL divergence is computed with respect to distribution of the latent
            #    variables obtained from the input of the graph (self.x_tilde)
            all_axis_but_first = list(range(len(gaussian_model_latent.batch_shape)))[1:]
            latent_loss = tf.reduce_sum(gaussian_model_latent.kl_divergence(prior), axis=all_axis_but_first)
            
            # average on the batch
            mean_latent_loss = tf.reduce_mean(latent_loss, axis=0, name="latent_loss")
            
        return mean_latent_loss
    '''
