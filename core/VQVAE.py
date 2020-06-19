import pdb

import numpy as np
import tensorflow as tf

from argo.core.hooks.LatentVarsGeometricClassificationHook import LatentVarsGeometricClassificationHook
from argo.core.network.AbstractAutoEncoder import AbstractAutoEncoder

from datasets.Dataset import TRAIN_LOOP, TRAIN, VALIDATION, TEST

from .ELBO import ELBO

from .LatentTraversalsHook import LatentTraversalsHook
from .PCALatentVariablesHook import PCALatentVariablesHook
from .TwoDimPCALatentVariablesHook import TwoDimPCALatentVariablesHook
from .AEImagesReconstructHook import AEImagesReconstructHook
from .VAELinearInterpolationHook import VAELinearInterpolationHook

from .VQVAENetwork import VQVAENetwork

from argo.core.CostFunctions import CostFunctions
from argo.core.hooks.FrechetInceptionDistanceHook import FrechetInceptionDistanceHook
from argo.core.hooks.ImagesGenerateHook import ImagesGenerateHook
from argo.core.hooks.ImportanceSamplingHook import ImportanceSamplingHook
from argo.core.hooks.LatentVarsClassificationHook import LatentVarsClassificationHook
from argo.core.hooks.LoggingMeanTensorsHook import LoggingMeanTensorsHook


class VQVAE(AbstractAutoEncoder):
    """ VQ - Variation Autoencoder

    """

    launchable_name = "VQVAE"

    default_params= {
        **AbstractAutoEncoder.default_params,

        "samples" : 10,

        "cost_function": ("VQELBO", {"beta" : 1.0,
                                   "warm_up_method" : None, # None | inverse_warm_up | warm_up
                                   }),

        "epochs" : 2000,                     # number of epochs
    }

    def create_id(self):

        _id = self.launchable_name

        # add to the ID the information of the cost function
        _id += '-c' + self._cost_function.create_id(self._opts["cost_function"][1])

        super_id = super().create_id()
        network_id = self._network.create_id()

        _id += super_id + network_id
        if "encoder" in self._opts["regularizers"] and len(self._opts["regularizers"]["encoder"].keys())>0:
            _id += '-crE' + self.create_custom_regularizers_id("encoder")
        if "decoder" in self._opts["regularizers"] and len(self._opts["regularizers"]["decoder"].keys())>0:
            _id += '-crD' + self.create_custom_regularizers_id("decoder")

        return _id

    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):

        # notice that in the following opts is used, and not self._opts, until the
        # parent constructor is called
        
        # NB need to create the network before the super init because id generation depends on the network
        self._network = VQVAENetwork(opts, "vqvae_network", seed=seed)
        self._cost_function = CostFunctions.instantiate_cost_function(opts["cost_function"], module_path= "vae")

        self.n_samples_prior = self._network.n_samples_prior

        # important nodes
        # self._approximate_posterior = None
        self._model_visible = None
        self.x_reconstruction_node = None
        self.samples = None

        super().__init__(opts, dirName, check_ops, gpu, seed)


    def create_network(self):
        
        # create autoencoder network
        out_net, net_losses = self._network(self.x, self.is_training)

        self.z_e = out_net['z_e']
        self.z_q = out_net['z_q']

        self.net_losses = net_losses

        self._prior = out_net["prior"]
        self._approximate_posterior = out_net["posterior"]

        self._model_visible = out_net["reconstruction_model"]
        self.x_reconstruction_node = tf.identity(self._model_visible.reconstruction_node(), name="reconstruction_node")

        self.samples = out_net["generation_node"]

        # 'perplexity': vq_out['perplexity'],
        # 'encodings': vq_out['encodings'],
        # 'encoding_indices': vq_out['encoding_indices'],
        # 'embeddings': self._embeddings,
        # 'reconstruction_model': self._model_visible,
        # 'generation_node': samples.reconstruction_node()
        #

    def create_loss(self):

        #TODO move up in AbstractAutoEncoder.
        # A TFDeepLearningModel model should be free to specify a model dependent loss..
        # compute the cost function, passing the model as a parameter
        self.loss, self.loss_nodes_to_log, self.loss_nodes_to_log_names, self.loss_nodes_to_log_filenames = self._cost_function(self)


    def encode(self, X, sess = None):
        """Encode data by mapping it into the latent space."""

        sess = sess or self.get_raw_session()

        return sess.run(self.z_q,
                         feed_dict={self.x: X})

    def decode(self, Z, sess = None, mask=None):
        """Decode latent vectors in input data."""
        sess = sess or self.get_raw_session()

        image = sess.run(self.x_reconstruction_node,
                         feed_dict={self.z_q: Z})

        return image

    def generate(self, batch_size=1, sess = None):
        """ Generate data by sampling from latent space.

        """
        sess = sess or self.get_raw_session()

        return sess.run(self.samples, feed_dict = {self.n_samples_prior : batch_size})

    def reconstruct(self, X, stp=False, sess=None):
        """ Use VAE to reconstruct given data. """
        sess = sess or self.get_raw_session()

        return sess.run(self.x_reconstruction_node,
                        feed_dict={self.x: X})


    def create_hooks(self, config):
        hooks = super().create_hooks(config)

        # LOGGING HOOKS
        tensors_to_average = self.loss_nodes_to_log
        tensors_to_average_names = self.loss_nodes_to_log_names
        tensors_to_average_plots = self.loss_nodes_to_log_filenames

        hooks.append(LoggingMeanTensorsHook(model=self,
                                            fileName="log",
                                            dirName=self.dirName,
                                            tensors_to_average=tensors_to_average,
                                            tensors_to_average_names=tensors_to_average_names,
                                            tensors_to_average_plots=tensors_to_average_plots,
                                            average_steps=self._n_steps_stats,
                                            tensorboard_dir=self._tensorboard_dir,
                                            trigger_summaries=config["save_summaries"],
                                            plot_offset=self._plot_offset,
                                            train_loop_key=TRAIN_LOOP,
                                            # if you want to remove some dataset from here, make support to specify from conf on which datasets to log, if in doubt ask me please. Riccardo
                                            datasets_keys=[TRAIN, VALIDATION, TEST],
                                            time_reference=self._time_reference_str
                                            )
                     )

        kwargs = config.get("ImagesReconstructHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(AEImagesReconstructHook(model=self,
                                                 dirName=self.dirName,
                                                 **kwargs)
                         )

        kwargs = config.get("ImagesGenerateHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(ImagesGenerateHook(model=self,
                                            dirName=self.dirName,
                                            **kwargs
                                            )
                         )

        # kwargs = config.get("ImportanceSamplingHook", None)
        # if kwargs:
        #     if not isinstance(kwargs, list):
        #         kwargs = [kwargs]
        #     for kw in kwargs:
        #         kws = {**self._plot_model_hooks_kwargs,
        #                **kw}
        #         hooks.append(ImportanceSamplingHook(model = self,
        #                                             dirName = self.dirName,
        #                                             tensors_to_average = [self.importance_sampling_node],
        #                                             datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
        #                                             **kws
        #                                            )
        #                     )
        #
        # kwargs = config.get("TwoDimPCALatentVariablesHook", None)
        # if kwargs:
        #     kwargs = {**self._default_model_hooks_kwargs,
        #               **kwargs}
        #     hooks.append(TwoDimPCALatentVariablesHook(model = self,
        #                                               dirName = self.dirName,
        #                                               tensors = [self.z] + list(self._approximate_posterior_params),
        #                                               tensors_names = ['z',
        #                                                                'mu'],
        #                                               datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
        #                                               **kwargs
        #                                              )
        #                  )
        #
        # kwargs = config.get("PCALatentVariablesHook", None)
        # if kwargs:
        #     kwargs = {**self._plot_model_hooks_kwargs,
        #               **kwargs}
        #     hooks.append(PCALatentVariablesHook(model = self,
        #                                          dirName = self.dirName,
        #                                          tensors = [self.z,
        #                                                     self._approximate_posterior_params[0]],
        #                                          tensors_names = ['z',
        #                                                           'mu'],
        #                                          datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
        #                                          **kwargs
        #                                        )
        #                  )
        #
        # kwargs = config.get("VAELinearInterpolationHook", None)
        # if kwargs:
        #     kwargs = {**self._default_model_hooks_kwargs,
        #               **kwargs}
        #     hooks.append(VAELinearInterpolationHook(model = self,
        #                                             dirName = self.dirName,
        #                                             **kwargs)
        #                  )
        #
        #
        # kwargs = config.get("LatentVarsClassificationHook", None)
        # if kwargs:
        #     kwargs = {**self._plot_model_hooks_kwargs,
        #                **kwargs}
        #     hooks.append(LatentVarsClassificationHook(model = self,
        #                                               dirName = self.dirName,
        #                                               tensors = [self.z,
        #                                                          self._approximate_posterior_params[0],
        #                                                          tf.concat(list(self._approximate_posterior_params), axis=1)],
        #                                               tensors_names = ['z',
        #                                                                'mu',
        #                                                                'mu_cov'],
        #                                               datasets_keys = [TRAIN,VALIDATION], # don't change the order (Luigi)
        #                                               **kwargs)
        #                   )
        #
        #
        # # frechet inception distance
        # kwargs = config.get("FrechetInceptionDistanceHook", None)
        # if kwargs:
        #     for kw in kwargs:
        #         kws = {**self._default_model_hooks_kwargs,
        #                **kw}
        #         hooks.append(FrechetInceptionDistanceHook(model = self,
        #                                                   dirName = self.dirName,
        #                                                   **kws)
        #                     )
        #
        # # latent traversals hook
        # kwargs = config.get("LatentTraversalsHook", None)
        # if kwargs:
        #     kwargs = {**self._default_model_hooks_kwargs,
        #               **kwargs}
        #     hooks.append(LatentTraversalsHook(model=self,
        #                                       dirName=self.dirName,
        #                                       **kwargs)
        #                                       )
        #
        # kwargs = config.get("LatentVarsGeometricClassificationHook", None)
        # if kwargs:
        #     kwargs = {**self._default_model_hooks_kwargs,
        #               **kwargs}
        #     hooks.append(LatentVarsGeometricClassificationHook(model=self,
        #                                                        dirName=self.dirName,
        #                                                        tensors=[self.z,
        #                                                                 self._approximate_posterior_params[0],
        #                                                                 tf.concat(list(self._approximate_posterior_params),
        #                                                                           axis=1)],
        #                                                        tensors_names=['z',
        #                                                                       'mu',
        #                                                                       'mu_cov'],
        #
        #                                                        datasets_keys=[TRAIN, VALIDATION, TEST],
        #                                                        **kwargs)
        #                  )

        return hooks

