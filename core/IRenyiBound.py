from .VAEFunction import VAEFunction
from tensorflow_probability import distributions as tfd

import tensorflow as tf

class IRenyiBound(VAEFunction):
    """ Class to implement Variational Renyi lower bound, as in Renyi Divergence
    Variational Inference paper, Li et al. (2016)
    Input:
         gaussian_model_observed : class of the model used for the output distribution
                                   of the generative network
         gaussian_model_latent   : class of the model for the output distribution of the
                                    encoder network
         binarized               : binary dataset flag
         synthetic               : synthetic dataset flag

    Methods:
         compute : return the Renyi lower bound
    """

    # TODOfunctionlogger: remove gaussian_model_latent and pass only what is needed (which most ikely is smaller)
    def __init__(self, gaussian_model_observed, gaussian_model_latent):
        VAEFunction.__init__(self)
        self._gaussian_model_observed = gaussian_model_observed
        self._gaussian_model_latent = gaussian_model_latent

    def compute(self, alpha, x_replicate, x_reconstruced_mean, z, h, normalize):
        """ implement Renyi cost function as in the paper Renyi Divergence Variational Inference,
            formula [5], with alpha in (0, 1)
            alpha = 0 => log p(x)
            alpha -> 1 => KL lower bound
        """
        '''
        # log conditional p(x|z)
        if self._binary==1:
            self.log_p_x_z = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                                                         labels=x_replicate,
                                                         logits=x_reconstructed_mean), 2)
        else:
            #assert self._synthetic==1
        '''

        self.log_p_x_z = self._gaussian_model_observed.log_pdf(x_replicate)
        self.log_p_x_z = tf.reshape(self.log_p_x_z, [-1, self._gaussian_model_latent._n_samples])

        '''
        if not self._synthetic:
            raise AssertionError("Renyi bound not yet implemented for continuous real dataset (e.g. MNISTcontinuos")
        '''
        
        # log posterior q(z|x)
        log_q_z_x = self._gaussian_model_latent.log_pdf(z)
        log_q_z_x = tf.reshape(log_q_z_x, [-1, self._gaussian_model_latent._n_samples])

        # log prior p(z)
        dim_z = self._gaussian_model_latent._dim
        log_p_z = tfd.MultivariateNormalDiag(tf.zeros(dim_z), tf.ones(dim_z))._log_prob(z)
        log_p_z = tf.reshape(log_p_z, [-1, self._gaussian_model_latent._n_samples])

        # exponent for Renyi expectation -- to avoid numerical issues we use exp of log
        # use log sum exp trick
        # TODOestimatefunction: recompute this using the tf function
        exponent = (1 - alpha) * (self.log_p_x_z + log_p_z - log_q_z_x)
        
        #max_exponent= tf.reduce_max(exponent, 1, keep_dims=True)
        #renyi_sum = tf.log(tf.reduce_mean(tf.exp(exponent - max_exponent), 1)) + tf.reduce_mean(max_exponent, 1)
        
        renyi_sum = -tf.reduce_logsumexp(exponent)/(1 - alpha)
        
        return renyi_sum

   
