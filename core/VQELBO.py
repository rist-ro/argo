import tensorflow as tf

from .ELBO import ELBO
from argo.core.utils.argo_utils import create_panels_lists

#from argo.core.utils.argo_utils import tf_clip

class VQELBO(ELBO):

    def __init__(self, use_kl=False, beta=1.0, warm_up_method=None, name="VQELBO"):
        super().__init__(beta=beta, warm_up_method=warm_up_method, name = name)
        self._use_kl = use_kl

    def create_id(self, cost_fuction_kwargs):
        _id = "VQELBO"
        if self._use_kl:
            _id += "_b" + str(cost_fuction_kwargs["beta"]) +\
                   self.create_warm_up_id(cost_fuction_kwargs)

        return _id

    def _build(self, vae):

        prior =  vae._prior
        approximate_posterior = vae._approximate_posterior

        vq_losses = vae.net_losses
        perplexity = vq_losses["perplexity"]
        vq_loss = vq_losses["vq_loss"]

        model_visible = vae._model_visible
        x_target =  vae.x_target
        n_z_samples = 1

        warm_up = self.get_warm_up_coefficient(self._warm_up_method, vae)

        rec_nll = self.reconstruction_loss(x_target, n_z_samples, model_visible, vae)

        KL = self.latent_loss(approximate_posterior, prior)

        # use kl only if specified, original VQ-VAE paper doesn't use
        kl_loss = warm_up * self.beta * KL if self._use_kl else 0.

        cost = kl_loss + vq_loss + rec_nll

        dim_with_channels = tf.cast(tf.reduce_prod(tf.shape(x_target)[1:]), tf.float32)

        # TODO I think this formula is still not correct
        # check https://www.reddit.com/r/MachineLearning/comments/56m5o2/discussion_calculation_of_bitsdims/
        bits_dim = (( rec_nll /dim_with_channels) + tf.log(256.0/2.0)) / tf.log(2.0)

        # First panel will be at screen during training
        list_of_vpanels_of_plots = [
            [
                    {
                        'nodes' : [cost],
                        'names': ["loss"],
                        'output': {'fileName': "loss"}
                    },

                    {
                        'nodes': [bits_dim],
                        'names': ["bd"],
                        'output': {'fileName': "bits_dim"}
                    },

            ],
            [
                {
                    'nodes': [KL],
                    'names': ["KL"],
                    'output': {'fileName': "kl"}
                },

                {
                    'nodes': [rec_nll],
                    'names': ["RL"],
                    'output': {'fileName': "rl"}
                },

                {
                    'nodes': [perplexity],
                    'names': ["perplexity"],
                    'output': {'fileName': "perplexity"}
                },

            ]
        ]

        nodes_to_log, names_of_nodes_to_log, filenames_to_log_to = create_panels_lists(list_of_vpanels_of_plots)

        return cost, nodes_to_log, names_of_nodes_to_log, filenames_to_log_to


    def latent_loss(self, approximate_posterior, prior):

        # two categorical distributions

        with tf.variable_scope('latent_loss'):
            mean_latent_loss = tf.reduce_mean(approximate_posterior.kl_divergence(prior), axis=0, name="latent_loss")

        return mean_latent_loss
