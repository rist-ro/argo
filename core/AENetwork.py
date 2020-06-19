from argo.core.network.ArgoStochasticNetworkWithDefaults import ArgoStochasticNetworkWithDefaults
from argo.core.network.GeneralSonnetNetwork import GeneralSonnetNetwork
from argo.core.utils import argo_utils as utils


# from .GaussianModelLatentVars import GaussianModelLatentVars
#from .GaussianDiagonal import GaussianDiagonal
#from .LogitNormalDiagonal import LogitNormalDiagonal
#from .NormalDiagonal import NormalDiagonal

# from .GaussianTridiagonalPrecision import GaussianTridiagonalPrecision
# from .GaussianBlockTridiagonalPrecisionTF import GaussianBlockTridiagonalPrecision
# from .GaussianRankOneUpdate import GaussianRankOneUpdate
# from .GaussianDiagonalZeroOne import GaussianDiagonalZeroOne
# from .GaussianObservedGridTf import GaussianObservedGrid
# from .GaussianIsotropic import GaussianIsotropic


class AENetwork(ArgoStochasticNetworkWithDefaults):
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
                ("Linear", {"output_size" : 100}, 1),
                ("Linear", {"output_size" : 50}, 1)],

            "decoder" : [
                ("Linear", {"output_size" : 100}, 1),
                ("Linear", {"output_size" : 200}, 1),
                ("Bernoulli", {},  0)]
        },
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

    def _build(self, x, is_training=False, network_str=""):
        """
        Args:
            x (tf.tensor): input node.
            network_str (str): Optional network_str specifying the network we are going to build.
                            It is used to set some specific collections for activity and contractive regularizers.

        Returns:
            h
            tf distribution on the visible neurons
        """

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

        self.h = self.encoder_module(x)

        # self._latent_vars_logger_class = self._gaussian_model_latent.get_vars_logger_class()

        # I set the shape of the visible layer, to match the input shape
        input_shape = x.get_shape().as_list()[1:]
        self._network_architecture["decoder"][-1][1]["output_shape"] = input_shape

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

        self._model_visible = self.decoder_module(self.h)

        return self.h, self._model_visible
