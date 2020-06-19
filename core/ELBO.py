import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule


#from argo.core.utils.argo_utils import tf_clip

class ELBO(AbstractModule):

    def __init__(self, beta=1.0, warm_up_method=None, name="ELBO", mask=None):
        super().__init__(name = name)

        self._warm_up_method = warm_up_method
        self.beta = tf.placeholder_with_default(float(beta), shape=(), name='beta_regularizer')

        assert(mask==None or mask==0 or mask==1)
        self.mask = mask
       
    
    def create_id(self, cost_fuction_kwargs):
        
        # cost_fuction_kwargs[0] is the cost function name

        _id = "ELBO_b" + str(cost_fuction_kwargs["beta"])
        _id += self.create_warm_up_id(cost_fuction_kwargs)
        if "mask" in cost_fuction_kwargs and cost_fuction_kwargs["beta"]==1:
            _id += "_m"

        return _id

    def create_warm_up_id(self, cost_fuction_kwargs):

        if "warm_up_method" not in cost_fuction_kwargs or cost_fuction_kwargs["warm_up_method"] is None or cost_fuction_kwargs["warm_up_method"][0] is None:
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
        if warm_up_method is None or warm_up_method[0] is None:
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
    
    def _build(self, vae):
        prior =  vae._prior
        approximate_posterior = vae._approximate_posterior
        model_visible = vae._model_visible
        x_target =  vae.x_target
        n_z_samples = vae.n_z_samples

        warm_up = self.get_warm_up_coefficient(self._warm_up_method, vae)

        # The loss is composed of two terms:
        #
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.

        reconstruction_loss = self.reconstruction_loss(x_target, n_z_samples, model_visible, vae)

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.


        KL, KL_i = self.latent_loss(approximate_posterior, prior)
        latent_loss = self.beta * KL

        cost = warm_up * latent_loss + reconstruction_loss

        # logging individual KL_i
        KL_i = tf.unstack(KL_i)
        KL_i_names = ["KL_" + str(int(i+1)) for i, l in enumerate(KL_i)]

        ELBO = -KL -reconstruction_loss
        
        return cost, [[ELBO], [-reconstruction_loss], [latent_loss], KL_i], [["ELBO"], ["RL"], ["KL"], KL_i_names], [{"fileName" : "cost_function_ELBO"}, {"fileName" : "cost_function_RL"}, {"fileName" : "cost_function_KL"}, {"fileName" : "cost_function_KL_i",  "legend": 0}]

    
    def reconstruction_loss(self, x_target, n_z_samples, model_visible, model=None):
        
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

            reconstr_loss = -model_visible.log_prob(x_replicate)
            #pdb.set_trace()
            
            #f.Print(self.mask, [self.mask])
            
            if self.mask == 1:
                # apply mask
                mask_replicate = tf.tile(model.mask, [n_z_samples] + ones)
                reconstr_loss = tf.multiply(reconstr_loss, mask_replicate)

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

            if len(mean_reconstr_loss.shape) > 0:
                raise RuntimeError("loss should be a scalar at this point, found shape: {}".format(mean_reconstr_loss.shape))

            # self.log_p_x_z = mean_reconstr_loss

        return mean_reconstr_loss
    
    def latent_loss(self, approximate_posterior, prior):
   
        # with tf.variable_scope('ELBO/reconstruction_loss'):
        # no need for ELBO, sonnet module is already adding that, the line above would produce:
        # ELBO/ELBO/reconstruction_loss/node_created
        with tf.variable_scope('latent_loss'):
            # 1) cast is required
            # 2) the KL divergence is computed with respect to distribution of the latent
            #    variables obtained from the input of the graph (self.x_tilde)
            #all_axis_but_first = list(range(len(gaussian_model_latent.batch_shape)))[1:]

            # average on the batch
            if len(approximate_posterior.kl_divergence(prior).shape)>1:
                # this is the case of the GaussianDiagonal
                mean_latent_loss_i = tf.reduce_mean(approximate_posterior.kl_divergence(prior), axis=0, name="latent_loss_i")
                mean_latent_loss = tf.reduce_sum(mean_latent_loss_i, name="latent_loss")
            else:
                # this is the case of the vMF        
                mean_latent_loss_i = tf.reduce_mean(approximate_posterior.kl_divergence(prior), axis=0, name="latent_loss_i")
                mean_latent_loss = mean_latent_loss_i
                mean_latent_loss_i = tf.reshape(mean_latent_loss, [1,])
                #mean_latent_loss_i = tf.reshape(mean_latent_loss_i, [1,-1])

        #pdb.set_trace()
                
        return mean_latent_loss, mean_latent_loss_i
