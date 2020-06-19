import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule

class HMLogLikelihood(AbstractModule):

    def __init__(self, name="HMLL"):
        super().__init__(name = name)

    def create_id(self, cost_fuction_kwargs):
        
        _id = "HMLL"

        return _id
    

    def _build(self, hm):
        n_z_samples = hm.n_z_samples


        x_target =  hm.x_target
        x_distr = hm._hgw[0][0]

        h_target = hm._prior_samples
        h_distr = hm._hrs[-1][0]

        # The loss is composed of two terms:
        #
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.

        reconstruction_loss = self.reconstruction_loss(x_target, n_z_samples, x_distr)
        dream_reconstruction_loss = self.reconstruction_loss(h_target, n_z_samples, h_distr)

        return None, [[reconstruction_loss], [dream_reconstruction_loss]], [["NLL_X"],["NLL_H"]], [{"fileName" : "reconstruction_loss_NLLX"},{"fileName" : "reconstruction_loss_NLLH"}]

    
    def reconstruction_loss(self, x_target, n_z_samples, distr):
        
        # with tf.variable_scope('LL/reconstruction_loss'):
        # no need for LL, sonnet module is already adding that, the line above would produce:
        # LL/LL/reconstruction_loss/node_created
        with tf.variable_scope('reconstruction_loss'):

            # 1) the log_pdf is computed with respect to distribution of the visible
            #    variables obtained from the target of input of the graph (self.x_target)
            # can I avoid replicate? maybe not..
            input_shape = x_target.shape.as_list()[1:]
            ones = [1] * len(input_shape)
            x_replicate = tf.tile(x_target, [n_z_samples] + ones)

            x_replicate = tf.reshape(x_replicate, [-1] + distr.logits.shape.as_list()[1:])

            reconstr_loss = - distr.log_prob(x_replicate)
            #now (ready for arbitrary intermediate samplings)
            all_axis_but_first = list(range(len(reconstr_loss.shape)))[1:]
            #independent p for each input pixel
            log_p = tf.reduce_sum(reconstr_loss, axis=all_axis_but_first)
            #average over all the samples and the batch (they are both stacked on the axis 0)
            mean_reconstr_loss = tf.reduce_mean(log_p, axis=0, name="reconstruction_loss")
            
        return mean_reconstr_loss

