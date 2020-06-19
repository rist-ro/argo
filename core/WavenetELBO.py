import tensorflow as tf

from .ELBO import ELBO
from .wavenet.IndependentChannelsTransformedDistribution import IndependentChannelsTransformedDistribution

LATENT_REGULARIZATION_TYPE_KL = 'KL'
LATENT_REGULARIZATION_TYPE_MMD = 'MMD'
LATENT_REGULARIZATION_TYPE_MUVAE = 'mu-VAE'
LATENT_REGULARIZATION_TYPE_SIGMA = 'sigma'


class WavenetELBO(ELBO):

    def __init__(self, beta=1.0, warm_up_method=None, latent_regularizations=((LATENT_REGULARIZATION_TYPE_KL, 1.0)),
                 name="WavenetELBO"):
        super().__init__(name=name, warm_up_method=warm_up_method, beta=beta)
        self.latent_regularizations = latent_regularizations

    def create_id(self, cost_fuction_kwargs):

        # cost_fuction_kwargs[0] is the cost function name
        regularization_str = ''
        at_least_one_warmup = False
        for latent_reg_params in self.latent_regularizations:
            latent_reg_type, weight, warmup = unpack_latent_reg_params(latent_reg_params)
            at_least_one_warmup = at_least_one_warmup or warmup
            regularization_str += '{:.1f}{}+'.format(weight, latent_reg_type)

        _id = "WELBO"
        if at_least_one_warmup:
            _id += self.create_warm_up_id(cost_fuction_kwargs)
        _id += '_' + regularization_str[:-1]

        return _id

    def _build(self, vae, isMAF=False):
        prior = vae._prior

        gaussian_model_latent = vae._gaussian_model_latent
        model_visible = vae._model_visible
        x_quant = vae.x_target_quantized

        x_indices = tf.cast(x_quant, tf.int32) + 128
        x_target = tf.squeeze(x_indices, axis=-1)

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

        reconstruction_loss = self.reconstruction_loss(x_target, n_z_samples, model_visible)

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.x
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.

        # KL, KL_i = self.kl(gaussian_model_latent, prior)
        latent_losses = []
        latent_loss_names = []
        # latent_losses_i = []
        for latent_reg_params in self.latent_regularizations:
            reg_type, weight, use_warmup = unpack_latent_reg_params(latent_reg_params)
            latent_loss, latent_loss_i = self._latent_loss(gaussian_model_latent, prior, vae.samples_posterior, reg_type)
            if use_warmup:
                latent_loss *= warm_up
            latent_losses.append(latent_loss * weight)
            latent_loss_names.append(reg_type)

        # KL, KL_i = self.latent_loss(gaussian_model_latent, prior)
        # latent_loss = self.beta * latent_loss

        cost = reconstruction_loss
        for latent_loss in latent_losses:
            cost += latent_loss

        # logging individual KL_i

        # latent_loss_shape = latent_loss_i.shape
        # if len(latent_loss_shape) > 1:
        #     all_but_last_axis = list(range(len(latent_loss_shape) - 1))
        #     latent_loss_i = tf.reduce_mean(latent_loss_i, axis=all_but_last_axis)
        #
        # latent_loss_i.set_shape((vae._network._latent_channels,))
        # latent_loss_i = tf.unstack(latent_loss_i)
        # # KL_i_names = ["KL_" + str(int(i + 1)) for i, l in enumerate(KL_i)]
        # KL_i_names = ['LATENT_{}'.format(i + 1) for i in range(vae._network._latent_channels)]

        return cost, [[-reconstruction_loss], latent_losses, ], [["RL"], latent_loss_names, ], \
               [
                   {"fileName": "cost_function_RL"},
                   {"fileName": "cost_function_LATENT"},
                   # {"fileName": "cost_function_LATENT_i", "legend": 0}
               ]

    def reconstr_loss_per_sample(self, x_target_t, n_z_samples, categorical_distribution):
        '''
        computes the log probability of x_target_t w.r.t the given categorical distribution model_visible
        Args:
            x_target_t: the audio signal with values in range [0, 255]
            n_z_samples: how many samples have been drawn from the latent distribution
            model_visible: the cateforical distribution at the end of the wavenet model

        Returns:

        '''
        with tf.variable_scope('reconstruction_loss_t'):
            # 1) the log_pdf is computed with respect to distribution of the visible
            #    variables obtained from the target of input of the graph (self.x_target)

            # can I avoid replicate? maybe not..
            input_shape = x_target_t.shape.as_list()[1:]
            ones = [1] * len(input_shape)
            x_replicate = tf.tile(x_target_t, [n_z_samples] + ones)

            reconstr_loss = -categorical_distribution.log_prob(x_replicate)

            # now (ready for arbitrary intermediate samplings)
            all_axis_but_first = list(range(len(reconstr_loss.shape)))[1:]
            # independent p for each input pixel
            reconstr_loss = tf.reduce_sum(reconstr_loss, axis=all_axis_but_first)

        return reconstr_loss

    def _latent_loss(self, approximate_posterior, prior, z_posterior, regularization_type):
        with tf.variable_scope('latent_loss'):
            if type(prior) is IndependentChannelsTransformedDistribution:
                return self.kl_aproximation_maf(approximate_posterior, prior)
            elif regularization_type == LATENT_REGULARIZATION_TYPE_MMD:
                return self.MMD(z_posterior, prior)
            elif regularization_type == LATENT_REGULARIZATION_TYPE_MUVAE:
                return self.mu_VAE(approximate_posterior, prior)
            elif regularization_type == LATENT_REGULARIZATION_TYPE_SIGMA:
                return self.sigma_reg(approximate_posterior)
            else:
                return self.kl(approximate_posterior, prior)

    def sigma_reg(self, approximate_posterior):
        sigma_sq = tf.matrix_diag_part(approximate_posterior.covariance())
        sigma_loss = (- tf.reduce_mean(tf.log(sigma_sq)) + tf.reduce_mean(sigma_sq) - 1) / 2
        sigma_loss_i = (- tf.reduce_mean(tf.log(sigma_sq), axis=-1) + tf.reduce_mean(sigma_sq, axis=-1) - 1) / 2

        return sigma_loss, sigma_loss_i


    def mu_VAE(self, approximate_posterior, prior):
        mean = approximate_posterior.mean()  # bs, ch, dim
        sigma_sq = tf.matrix_diag_part(approximate_posterior.covariance())  # bs, ch, dim
        mmd_kl = (tf.abs(tf.reduce_mean(mean)) - tf.reduce_mean(tf.log(sigma_sq)) + tf.reduce_mean(sigma_sq) - 1) / 2
        mmd_kl_i = (tf.abs(tf.reduce_mean(mean, axis=-1)) - tf.reduce_mean(tf.log(sigma_sq), axis=-1)
                    + tf.reduce_mean(sigma_sq, axis=-1) - 1) / 2
        return mmd_kl, mmd_kl_i

    def MMD(self, q_z, prior):
        batch_size = tf.shape(q_z)[0]
        p_z = prior.sample(batch_size)

        p_kernel, p_kernel_i = gaussian_kernel(p_z, p_z)
        q_kernel, q_kernel_i = gaussian_kernel(q_z, q_z)
        p_q_kernel, p_q_kernel_i = gaussian_kernel(q_z, p_z)

        len_shape = len(p_kernel_i.get_shape().as_list())
        all_axis_but_last = list(range(len_shape - 1))

        mmd = tf.reduce_mean(p_kernel) + tf.reduce_mean(q_kernel) - 2 * tf.reduce_mean(p_q_kernel)
        # mmd_i = tf.reduce_mean(p_kernel, axis=all_axis_but_last) + tf.reduce_mean(q_kernel, axis=all_axis_but_last)\
        #         - 2 * tf.reduce_mean(p_q_kernel, all_axis_but_last)
        mmd_i = p_kernel_i + q_kernel_i - 2 * p_q_kernel_i

        return mmd, mmd_i

    def kl_aproximation_maf(self, approximate_posterior, prior):
        # with tf.variable_scope('latent_loss'):
        posterior_samples = approximate_posterior.sample(50)
        kl_posterior_prior = approximate_posterior.log_prob(posterior_samples) - prior.log_prob(posterior_samples)
        mean_latent_loss_i = tf.reduce_mean(kl_posterior_prior, axis=0, name='latent_loss_i')
        mean_latent_loss = tf.reduce_sum(mean_latent_loss_i, name="latent_loss")

        return mean_latent_loss, mean_latent_loss_i

    def kl(self, approximate_posterior, prior):

        # with tf.variable_scope('ELBO/reconstruction_loss'):
        # no need for ELBO, sonnet module is already adding that, the line above would produce:
        # ELBO/ELBO/reconstruction_loss/node_created
        # with tf.variable_scope('latent_loss'):
        # 1) cast is required
        # 2) the KL divergence is computed with respect to distribution of the latent
        #    variables obtained from the input of the graph (self.x_tilde)
        # all_axis_but_first = list(range(len(gaussian_model_latent.batch_shape)))[1:
        # average on the batch
        kl_posterior_prior = approximate_posterior.kl_divergence(prior)
        # kl_posterior_prior = tf.Print(kl_posterior_prior, [tf.shape(kl_posterior_prior)], '\n\nShape kl\n\n')

        if len(kl_posterior_prior.shape) > 1:
            # this is the case of the GaussianDiagonal
            mean_latent_loss_i = tf.reduce_mean(kl_posterior_prior, axis=0,
                                                name="latent_loss_i")
            mean_latent_loss = tf.reduce_sum(mean_latent_loss_i, name="latent_loss")
        else:
            # this is the case of the vMF
            mean_latent_loss_i = tf.reduce_mean(kl_posterior_prior, axis=0,
                                                name="latent_loss_i")
            mean_latent_loss = mean_latent_loss_i
            mean_latent_loss_i = tf.reshape(mean_latent_loss, [1, ])
            # mean_latent_loss_i = tf.reshape(mean_latent_loss_i, [1,-1])

        return mean_latent_loss, mean_latent_loss_i


def gaussian_kernel(x, y):  # [batch_size, ch, z_dim], [batch_size, ch, z_dim]
    x_size = tf.shape(x)[0]  # xbs
    y_size = tf.shape(y)[0]  # ybs
    dim = tf.shape(x)[2]  # zdim
    ch = tf.shape(x)[1]

    # straight forward method
    # exponent = - (x-y)**2 / tf.cast((2 * dim * ch), tf.float32)
    # kernel = tf.exp(exponent)
    # return tf.reduce_mean(kernel, axis=[1, 2]), tf.reduce_mean(kernel, axis=[2])

    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, ch, dim])), tf.stack([1, y_size, 1, 1]))  # xbs, ybs, dim
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, ch, dim])), tf.stack([x_size, 1, 1, 1]))

    kernel = tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=[2, 3]) / tf.cast(dim * ch, tf.float32))  # bs bs
    # per channel:
    kernel_i = tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=[3]) / tf.cast(dim, tf.float32))  # bs bs ch
    return kernel, kernel_i

def unpack_latent_reg_params(latent_reg_params):
    return latent_reg_params['type'], latent_reg_params['weight'], latent_reg_params['use_warmup']
