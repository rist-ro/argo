from .VAECostFunction import VAECostFunction

import pdb

import tensorflow as tf

class KLCostFunction(VAECostFunction):

    def __init__(self, binarized, gaussian_model_observed, gaussian_model_latent):
        VAECostFunction.__init__(self, binarized)

        self._gaussian_model_observed = gaussian_model_observed
        self._gaussian_model_latent = gaussian_model_latent
        
        self._cost_logger_class = None

    def compute(self, x, x_replicate, x_reconstr_mean_samples):
        # The loss is composed of two terms:
        #
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.

        if self._binarized==1:
            reconstr_loss = tf.reduce_sum(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                         labels=x_replicate, 
                                                         logits=x_reconstr_mean_samples), 1), 1)
           
        else:
            # see formula from https://en.wikipedia.org/wiki/Logit-normal_distribution
            x_logit = tf.log(x_replicate) - tf.log(1 - x_replicate)
            self.Bsum = tf.reduce_sum(tf.log(x_replicate) + tf.log(1 - x_replicate), axis=1)
            # here I have -log p(x), since I will maximize this quantity
            reconstr_loss = self.Bsum - self._gaussian_model_observed.log_pdf(x_logit)
    
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.

        latent_loss = self._gaussian_model_latent.kl_divergence_from_unit_covariance()

        '''
        if self.k==-1:
            latent_loss = -0.5 * tf.add(tf.reduce_sum(1 + 2 * self.z_chol_diag
                                                     - tf.square(self.z_mean)
                                                     - tf.exp(2 * self.z_chol_diag), 1),
                                        -tf.reduce_sum(tf.square(self.z_chol_nondiag), 1))

        '''

        return reconstr_loss + latent_loss




