import tensorflow as tf
from tensorflow_probability import distributions as tfd

from argo.core.utils.argo_utils import NUMTOL

from .ELBO import ELBO

class IELBO(ELBO):

    def __init__(self, beta=1.0, warm_up_method=None, h=0.01, normalize=1, name = "IELBO"):
        super().__init__(beta=beta, warm_up_method=warm_up_method, name = name)

        self._h = h
        self._normalize = normalize

    def create_id(self, cost_fuction_kwargs):
        _id = "IELBO_b" + str(cost_fuction_kwargs["beta"]) + "_h" + str(cost_fuction_kwargs["h"])  + "_n" + str(cost_fuction_kwargs["normalize"])

        _id += self.create_warm_up_id(cost_fuction_kwargs)
        
        return _id
        
    def _build(self, vae):

        self.h = tf.placeholder_with_default(self._h, shape=(), name='half_intergral_interval')
        self.normalize = tf.placeholder_with_default(self._normalize, shape=(), name='normalize_integral')

        assert(not isinstance(vae._model_visible, tfd.Bernoulli)), "Cannot use the IELBO with discrete distributions for the visible variables"

        return super()._build(vae)

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

            # no need to check for the values in the interval, since the cdf is defined over R
            upper = model_visible.cdf(x_replicate + self.h)
            lower = model_visible.cdf(x_replicate - self.h)

            delta = tf.cond(tf.equal(self.normalize, 1),
                            lambda: 2.0 * self.h + NUMTOL,
                            lambda: 1.0)
            
            reconstr_loss =  -tf.log( (upper-lower) / delta + NUMTOL)
            
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
