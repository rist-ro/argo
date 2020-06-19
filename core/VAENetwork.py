import tensorflow as tf

from argo.core.network.ArgoStochasticNetworkWithDefaults import ArgoStochasticNetworkWithDefaults
from argo.core.network.GeneralSonnetNetwork import GeneralSonnetNetwork
from argo.core.utils import argo_utils as utils


class VAENetwork(ArgoStochasticNetworkWithDefaults):
    """
    Network class for managing a VAE network
    """

    # if not present in opts, the values specified in default_params will be provided automagically
    # see more in AbstractNetwork.__init__
    default_params = {
        **ArgoStochasticNetworkWithDefaults.default_params,
        "network_architecture" : {
            "encoder" : [
                ("Linear", {"output_size" : 200}, 1),
                ("Linear", {"output_size" : 200}, 1),
                ("GaussianDiagonal", {"size" : 20, "minimal_covariance" : 0}, 0)],

            "decoder" : [
                ("Linear", {"output_size" : 200}, 1),
                ("Linear", {"output_size" : 200}, 1),
                ("GaussianDiagonalZeroOne", {"minimal_covariance" : 0}, 0)],
        },

        "samples" : 10,
    }

    def create_id(self):

        encoder_layers_ids = filter(None,
                                    [utils.get_method_id(layer_tuple)
                                        for layer_tuple in self._opts["network_architecture"]["encoder"]]
                             )

        decoder_layers_ids = filter(None,
                                    [utils.get_method_id(layer_tuple)
                                        for layer_tuple in self._opts["network_architecture"]["decoder"]]
                              )

        _id = '-ne_' + "_".join(encoder_layers_ids) + \
              '-nd_' + "_".join(decoder_layers_ids)


        super_id = super().create_id()

        _id += super_id
        return _id


    def __init__(self, opts, name="vae_network", seed=None):
        """Short summary.

        Args:
            opts (dict): parameters of the task.
            name (str): name of the Sonnet module.
        """

        super().__init__(opts, name, seed)

        network_architecture = self._opts["network_architecture"]
        # TODO-ARGO2 you might want to check here that the 2 architectures (encoder and decoder) expected are effectively passed
        self._network_architecture = network_architecture

        self.n_z_samples = tf.placeholder_with_default(self._opts["samples"], shape=(), name='n_z_samples')

    def _build(self, x, is_training=False, network_str=""):
        """
        Args:
            x (tf.tensor): input node.
            network_str (str): Optional network_str specifying the network we are going to build.
                            It is used to set some specific collections for activity and contractive regularizers.

        Returns:
            tf distribution of the latent space
            tf distribution on the visible neurons
        """

        # The create_graph function creates the graph which consists of 5 parts:
        # 1) the recognition network
        # 2) the model for the latent variables
        # 3) the sampling from the output of the recognition
        # 4) the generative network which takes the sampled points as input
        # 5) the likelihood model for the visible variables

        # 1) and 2)
        # import pdb;pdb.set_trace()
        self.encoder_module = GeneralSonnetNetwork(self._network_defaults["activation"],
                                                   self._network_defaults["weights_init"],
                                                   self._network_defaults["bias_init"],
                                                   self._network_defaults["weights_reg"],
                                                   self._network_defaults["bias_reg"],
                                                   self._network_architecture["encoder"],
                                                   self._stochastic_defaults,
                                                   is_training=is_training,
                                                   network_str=network_str,
                                                   name="encoder")
        self._approximate_posterior = self.encoder_module(x)
        # self._latent_vars_logger_class = self._gaussian_model_latent.get_vars_logger_class()

        # 3)
        with tf.variable_scope('latent_sampling'):
            # ISSUE
            #batch_reshape = [-1] + self._approximate_posterior.batch_shape.as_list()[1:]
            if len(self._approximate_posterior.event_shape.as_list())==0:
                # Gaussian
                batch_reshape = [-1] + self._approximate_posterior.batch_shape.as_list()[1:]
            else:
                # von Miser-Fisher
                batch_reshape = [-1] + self._approximate_posterior.event_shape.as_list()
            self.z = tf.reshape(self._approximate_posterior.sample(self.n_z_samples), batch_reshape)
            #pdb.set_trace()
            self.log_pdf_z = self._approximate_posterior.log_prob(self.z)

        # I set the shape of the visible layer, to match the input shape
        input_shape = x.get_shape().as_list()[1:]
        self._network_architecture["decoder"][-1][1]["output_shape"] = input_shape

        # 4) and 5)
        self.decoder_module = GeneralSonnetNetwork(self._network_defaults["activation"],
                                                   self._network_defaults["weights_init"],
                                                   self._network_defaults["bias_init"],
                                                   self._network_defaults["weights_reg"],
                                                   self._network_defaults["bias_reg"],
                                                   self._network_architecture["decoder"],
                                                   self._stochastic_defaults,
                                                   is_training=is_training,
                                                   network_str=network_str,
                                                   name="decoder")

        self._model_visible = self.decoder_module(self.z)

        # create prior
        prior_shape = self._approximate_posterior.params()[0].shape.as_list()[1:]
        prior = self._approximate_posterior.default_prior(prior_shape)
        prior_samples = prior.sample(self.n_z_samples)
        
        # attach the decoder to the samples
        samples = self.decoder_module(prior_samples)
        
        return self._approximate_posterior, self.z, self._model_visible, prior, samples.reconstruction_node()
        
        
