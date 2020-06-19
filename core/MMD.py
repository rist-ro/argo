import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule


class MMD(AbstractModule):

    def __init__(self, mmd_kernel, beta=1.0, warm_up_method=None, name="MMD"):
        """
        Maximum mean discrepancy-based penalty used in Wasserstein AEs
        official implementation: https://github.com/tolstikhin/wae

        :param mmd_kernel: string, one of ["RBF", "IMQ"]; kernel to be used for computing MMD penalty ()
        :param beta: float; MMD coefficient
        :param warm_up_method: string, one of None, warm_up, inverse_warm_up
        :param name: string; a name for the loss
        """
        super().__init__(name=name)

        self._warm_up_method = warm_up_method
        self.beta = tf.placeholder_with_default(float(beta), shape=(), name='beta_regularizer')
        self.mmd_kernel = mmd_kernel

        self._check_kernel()

    def create_id(self, cost_fuction_kwargs):
        _id = "MMD_b" + str(cost_fuction_kwargs["beta"]) + "_" + self.mmd_kernel
        _id += self.create_warm_up_id(cost_fuction_kwargs)

        return _id

    def _check_kernel(self):
        if self.mmd_kernel not in ['RBF', 'IMQ']:
            raise ValueError("mmd_kernel not recognized")

    # duplicate code here (taken from ELBO) we may want to use warm-up to anneal the MMD penalty
    def create_warm_up_id(self, cost_fuction_kwargs):

        if "warm_up_method" not in cost_fuction_kwargs or cost_fuction_kwargs["warm_up_method"] is None or \
                cost_fuction_kwargs["warm_up_method"][0] is None:
            warm_up = "0"
        elif cost_fuction_kwargs["warm_up_method"][0] == "warm_up":
            warm_up = "W" + str(cost_fuction_kwargs["warm_up_method"][1]["epoch"])
        elif cost_fuction_kwargs["warm_up_method"][0] == "inverse_warm_up":
            warm_up = "IW" + str(cost_fuction_kwargs["warm_up_method"][1]["epoch"])
        else:
            raise ValueError("warm_up_method not recognized")
        _id = '_wu' + warm_up

        return _id

    def get_warm_up_coefficient(self, warm_up_method, vae_model):
        # Warm up: gradually introduce the latent loss in the cost function,
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
                                      tf.constant(warm_up_final_step - 1, dtype=tf.float32)
                              )

        elif warm_up_method[0] == "inverse_warm_up":

            # TODO-LUIGI what is below needs to be reimplemented
            raise Exception("what is below needs to be reimplemented")
            warm_up_final_step = warm_up_method[1]["epoch"] * vae_model.n_batches_per_epoch

            # decrease the KL scale from 10 (epoch 0) to 1 (epoch warm_up_method[1]["epoch"])
            # in general the formula is n - ((n-1)*epoch) / warm_up_method[1]["epoch"]
            warm_up = tf.constant(10, dtype=tf.float32) - tf.cast(9 * global_epoch, dtype=tf.float32) / tf.constant(
                float(warm_up_method[1]["epoch"]), dtype=tf.float32)

        else:
            raise ValueError("warm_up_method not recognized")

        return warm_up

    def _build(self, vae):
        prior = vae._prior
        gaussian_model_latent = vae._gaussian_model_latent
        x_target = vae.x_target
        reconstructions = vae.x_reconstruction_node
        batch_size = tf.shape(reconstructions)[0]

        warm_up = self.get_warm_up_coefficient(self._warm_up_method, vae)

        # The loss is composed of two terms:
        #
        # 1.) The reconstruction loss (can be any dissimilarity in the data space).

        reconstruction_loss = self.reconstruction_loss(x_target, reconstructions)

        # 2.) The latent loss, which is defined as the MMD
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.

        # should choose between the mean or z, if we use deterministic or stochastic encoders
        # for now, the encoder is stochastic
        mmd = self.latent_mmd_loss(gaussian_model_latent.sample(), prior.sample(batch_size))
        # mmd = self.latent_mmd_loss(gaussian_model_latent.mean(), prior.sample(batch_size))

        latent_loss = self.beta * mmd

        cost = warm_up * latent_loss + reconstruction_loss

        return cost, [[reconstruction_loss], [latent_loss]], [["RL"], ["MMD"]], [{"fileName": "cost_function_RL"}, {"fileName": "cost_function_MMD"}]

    def reconstruction_loss(self, x_target, reconstruction):

        with tf.variable_scope('reconstruction_loss'):
            num_dims = len(reconstruction.get_shape().as_list()) - 1
            loss = tf.reduce_sum(tf.square(x_target - reconstruction), axis=list(range(1, num_dims + 1)))
            mean_reconstr_loss = 0.05 * tf.reduce_mean(loss)

            if len(mean_reconstr_loss.shape) > 0:
                raise RuntimeError(
                    "loss should be a scalar at this point, found shape: {}".format(mean_reconstr_loss.shape))

        return mean_reconstr_loss

    def latent_mmd_loss(self, samples_qz, samples_pz):
        sigma2_p = 1
        kernel = self.mmd_kernel
        n = tf.shape(samples_qz)[0]
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) // 2
        z_dim = tf.shape(samples_qz)[1]

        norms_pz = tf.reduce_sum(tf.square(samples_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(samples_pz, samples_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(samples_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(samples_qz, samples_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(samples_qz, samples_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            res1 = tf.exp(- distances_qz / 2. / sigma2_k)
            res1 += tf.exp(- distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp(- distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

            c_base = 2. * tf.cast(z_dim, tf.float32) * sigma2_p
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = c_base * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat

