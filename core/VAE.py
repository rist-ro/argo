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
from .VAEImagesReconstructHook import VAEImagesReconstructHook
from .VAELinearInterpolationHook import VAELinearInterpolationHook

from .VAENetwork import VAENetwork

from argo.core.CostFunctions import CostFunctions
from argo.core.hooks.FrechetInceptionDistanceHook import FrechetInceptionDistanceHook
from argo.core.hooks.ImagesGenerateHook import ImagesGenerateHook
from argo.core.hooks.ImportanceSamplingHook import ImportanceSamplingHook
from argo.core.hooks.LatentVarsClassificationHook import LatentVarsClassificationHook
from argo.core.hooks.LoggingMeanTensorsHook import LoggingMeanTensorsHook


class VAE(AbstractAutoEncoder):
    """ Variation Autoencoder (VAE)
    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.

    """

    launchable_name = "VAE"

    default_params= {
        **AbstractAutoEncoder.default_params,

        # IGNORED
        # "minimal_covariance_z" : 0, # value added to the guassian diagonal entries of p(z|x)
        # "minimal_covariance_x" : 0, # value added to the guassian diagonal entries of p(x|z)
        # INSTEAD see the "minimal_covariance" attribute for the GaussianDiagonal,
        # GaussianDiagonalZeroOne and LogitNormal layers
        # default values are 0.0

        "samples" : 10,

        "cost_function": ("ELBO", {"beta" : 1.0,
                                   "warm_up_method" : None, # None | inverse_warm_up | warm_up
                                   }),

        "epochs" : 2000,                     # number of epochs
    }

    def create_id(self):

        _id = self.launchable_name

        # add to the ID the information of the cost function
        _id += '-c' + self._cost_function.create_id(self._opts["cost_function"][1])

        _id += '-s' + str(self._opts["samples"])
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
        self._network = VAENetwork(opts, "vae_network", seed=seed)
        self._cost_function = CostFunctions.instantiate_cost_function(opts["cost_function"], module_path= "vae")

        self.n_z_samples = self._network.n_z_samples

        # # replicate x
        # self.x_replicate = {}
        # self.x_target_replicate = {}

        # important nodes
        self._approximate_posterior = None
        self._model_visible = None
        self.x_reconstruction_node = None
        self.samples = None

        self.mask = None
        
        super().__init__(opts, dirName, check_ops, gpu, seed)

    def create_hooks(self, config):
        hooks = super().create_hooks(config)

        #LOGGING HOOKS

        # TODO here I am ignoring the number of channels, is it correct? (Luigi).
        # TODO The channels seem to be included with np.prod...
        # dim_with_channels = np.prod(self.x_shape["train"])

        dim_with_channels = tf.cast(tf.reduce_prod(tf.shape(self.raw_x)[1:]), tf.float32)
        # TODO the two shapes could be different, maybe use tf.shape(self.raw_x) (Riccardo)
        # dim_with_channels = np.prod(self.x_shape["train"])
        # dim_with_channels = np.prod(self.x_shape["eval"])

        # TODO I think this formula is still not correct, it is using loss (which inside has beta and warm_up), it is more correct than the one before maybe
        # check https://www.reddit.com/r/MachineLearning/comments/56m5o2/discussion_calculation_of_bitsdims/
        bits_dim = ((self.loss/dim_with_channels) + tf.log(256.0/2.0)) / tf.log(2.0)

        tensors_to_average = [
         [[ -self.loss ],
          [ bits_dim ]
         ],
          self.loss_nodes_to_log
        ]
        tensors_to_average_names = [
         [[ "LB_log(p)" ],
          [ "b/d" ]
         ],
          self.loss_nodes_to_log_names
        ]
        tensors_to_average_plots = [
         [{"fileName" : "loss"},
          {"fileName" : "bits_dim"}
         ],
          self.loss_nodes_to_log_filenames
        ]
        #[*[name for name in self.loss_nodes_to_track]],

        hooks.append(LoggingMeanTensorsHook(model = self,
                                            fileName = "log",
                                            dirName = self.dirName,
                                            tensors_to_average = tensors_to_average,
                                            tensors_to_average_names = tensors_to_average_names,
                                            tensors_to_average_plots = tensors_to_average_plots,
                                            average_steps=self._n_steps_stats,
                                            tensorboard_dir = self._tensorboard_dir,
                                            trigger_summaries = config["save_summaries"],
                                            plot_offset = self._plot_offset,
                                            train_loop_key = TRAIN_LOOP,
                                            # if you want to remove some dataset from here, make support to specify from conf on which datasets to log, if in doubt ask me please. Riccardo
                                            datasets_keys = [TRAIN, VALIDATION, TEST],
                                            time_reference = self._time_reference_str
                                            )
                     )
        

        kwargs = config.get("ImagesReconstructHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(VAEImagesReconstructHook(model = self,
                                                  dirName = self.dirName,
                                                  **kwargs)
                         )

        kwargs = config.get("ImagesGenerateHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(ImagesGenerateHook(model = self,
                                           dirName = self.dirName,
                                           **kwargs
                                           )
                         )

        kwargs = config.get("ImportanceSamplingHook", None)
        if kwargs:
            # create the IS node only when needed, this function fails when z is fully convolutional... (Riccardo)
            self.importance_sampling_node = self._create_importance_sampling_node()

            if not isinstance(kwargs, list):
                kwargs = [kwargs]
            for kw in kwargs:
                kws = {**self._plot_model_hooks_kwargs,
                       **kw}
                hooks.append(ImportanceSamplingHook(model = self,
                                                    dirName = self.dirName,
                                                    tensors_to_average = [self.importance_sampling_node],
                                                    datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
                                                    **kws
                                                   )
                            )

        kwargs = config.get("TwoDimPCALatentVariablesHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(TwoDimPCALatentVariablesHook(model = self,
                                                      dirName = self.dirName,
                                                      tensors = [self.z] + list(self._approximate_posterior_params),
                                                      tensors_names = ['z',
                                                                       'mu'],
                                                      datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
                                                      **kwargs
                                                     )
                         )

        kwargs = config.get("PCALatentVariablesHook", None)
        if kwargs:
            kwargs = {**self._plot_model_hooks_kwargs,
                      **kwargs}
            hooks.append(PCALatentVariablesHook(model = self,
                                                 dirName = self.dirName,
                                                 tensors = [self.z,
                                                            self._approximate_posterior_params[0]],
                                                 tensors_names = ['z',
                                                                  'mu'],
                                                 datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
                                                 **kwargs
                                               )
                         )

        kwargs = config.get("VAELinearInterpolationHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(VAELinearInterpolationHook(model = self,
                                                    dirName = self.dirName,
                                                    **kwargs)
                         )


        kwargs = config.get("LatentVarsClassificationHook", None)
        if kwargs:
            kwargs = {**self._plot_model_hooks_kwargs,
                       **kwargs}
            hooks.append(LatentVarsClassificationHook(model = self,
                                                      dirName = self.dirName,
                                                      tensors = [self.z,
                                                                 self._approximate_posterior_params[0],
                                                                 tf.concat(list(self._approximate_posterior_params), axis=1)],
                                                      tensors_names = ['z',
                                                                       'mu',
                                                                       'mu_cov'],
                                                      datasets_keys = [TRAIN,VALIDATION], # don't change the order (Luigi)
                                                      **kwargs)
                          )


        # frechet inception distance
        kwargs = config.get("FrechetInceptionDistanceHook", None)
        if kwargs:
            for kw in kwargs:
                kws = {**self._default_model_hooks_kwargs,
                       **kw}
                hooks.append(FrechetInceptionDistanceHook(model = self,
                                                          dirName = self.dirName,
                                                          **kws)
                            )

        # latent traversals hook
        kwargs = config.get("LatentTraversalsHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(LatentTraversalsHook(model=self,
                                              dirName=self.dirName,
                                              **kwargs)
                                              )

        kwargs = config.get("LatentVarsGeometricClassificationHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(LatentVarsGeometricClassificationHook(model=self,
                                                               dirName=self.dirName,
                                                               tensors=[self.z,
                                                                        self._approximate_posterior_params[0],
                                                                        tf.concat(list(self._approximate_posterior_params),
                                                                                  axis=1)],
                                                               tensors_names=['z',
                                                                              'mu',
                                                                              'mu_cov'],

                                                               datasets_keys=[TRAIN, VALIDATION, TEST],
                                                               **kwargs)
                         )

        return hooks

    def create_input_nodes(self, dataset):

        super(VAE, self).create_input_nodes(dataset)

        # check if the dataset specifies a mask (Alexandra); so far, implemented only in ELBO
        if isinstance(self._cost_function, ELBO) and hasattr(dataset, 'use_mask'):

            init_dims = [0] + [0 for s in self.x.shape[1:-1].as_list()]
            end_dims = [-1] + [-1 for s in self.x.shape[1:-1].as_list()]

            # take it from datasets_nodes[0][0] so that no perturbation is applied
            # and raw_x can be modified by removing the mask
            # see create_datasets_with_handles in TFDeepLearningModel.py (Luigi)
            self.mask = tf.slice(self.datasets_nodes[0][0], init_dims + [0], end_dims + [1])
            self.x = tf.slice(self.x, init_dims + [1], end_dims + [-1])
            
            self.raw_x = tf.slice(self.raw_x, init_dims + [1], end_dims + [-1])
            self.x_target = tf.slice(self.x_target, init_dims + [1], end_dims + [-1])

            
    def create_network(self):
        
        # create autoencoder network
        self._approximate_posterior, self.z, self._model_visible, self._prior, self.samples = self._network(self.x, is_training=self.is_training)

        self.x_reconstruction_node = tf.identity(self._model_visible.reconstruction_node(), name="reconstruction_node")
        # NB: I need to create the node now, otherwise the call to mean() will generate an error
        #pdb.set_trace()
        
        # since the graph will be finalized
        
        # old implemenation, depending on the GaussianDiagonal in the latent
        #self._gaussian_model_latent_mean = self._gaussian_model_latent.mean()
        #self._gaussian_model_latent_cov = tf.square(self._gaussian_model_latent.stddev())

        # new method
        self._approximate_posterior_params = self._approximate_posterior.params()

        # TODO this is not compatible with arbitrary distributions in output,
        # TODO maybe we can find a better solution also for reconstruction_node if we start having several needs..
        # reconstruction_node_mean = tf.identity(self._model_visible.mean(), name="reconstruction_mean")
        # nodes_to_be_saved = [tf.identity(node, name="model_visible_parameter_" + str(i)) for i, node in enumerate(self._model_visible.distribution_parameters())]
        # self._pb_output_nodes = ["reconstruction_node"] + ["model_visible_parameter_" + str(i) for i in range(len(self._model_visible.distribution_parameters()))]


    def _create_importance_sampling_node(self):

        # create the node for the importance sampling estimator of lop p
        # ISSUE
        #pdb.set_trace()
        #dim_z = self._approximate_posterior.batch_shape.as_list()[1]

        dim_x = np.prod(self._model_visible.batch_shape.as_list()[1:])
            
        if len(self._approximate_posterior.event_shape.as_list())==0:
            # Gaussian Diagonal           
            dim_z = self._approximate_posterior.batch_shape.as_list()[1]

            z_reshaped = tf.reshape(self.z, [self.n_z_samples, -1, dim_z])
            log_pdf_latent_reshaped = self._approximate_posterior.log_prob(z_reshaped)

        
        else:
            # von Mises

            # differently from th Gaussian, we have only one pdf
            # (no factorization over the latent variable) thus we can force in the reshape
            # dim_z = 1
            dim_z = 1

            #pdb.set_trace()
            
            #log_pdf_latent_reshaped = tf.reshape(self._approximate_posterior.log_prob(self.z), [self.n_z_samples, -1, dim_z])

            # fixed da Norbert, previosly it was 
            # z_reshaped = tf.reshape(self.z, [self.n_z_samples, -1])
            dim_z_of_prior = self._approximate_posterior.event_shape.as_list()[0]
            z_reshaped = tf.reshape(self.z, [self.n_z_samples, -1, dim_z_of_prior])
  
            
            log_pdf_latent_reshaped = self._approximate_posterior.log_prob(z_reshaped)
            log_pdf_latent_reshaped = tf.reshape(log_pdf_latent_reshaped, [self.n_z_samples, -1, dim_z])

        # here the size is [n_z_samples x batch_size , dim_z]
        log_pdf_prior = self._prior.log_prob(self.z)
        log_pdf_prior_reshaped = tf.reshape(log_pdf_prior, [self.n_z_samples, -1, dim_z])

        
        # can I avoid replicate? maybe not..
        input_shape = self.x.shape.as_list()[1:]
        ones = [1] * len(input_shape)
        x_replicate = tf.tile(self.x, [self.n_z_samples] + ones)

        log_pdf_conditional = self._model_visible.log_prob(x_replicate)
        log_pdf_conditional_reshaped = tf.reshape(log_pdf_conditional, [self.n_z_samples, -1, dim_x])

        #log_pdf_latent = tf.reshape(log_pdf_latent_reshaped, [-1, dim_z])

        # ADD HERE
        #log_pdf_latent_reshaped = tf.reshape(log_pdf_latent_reshaped, tf.shape(log_pdf_prior_reshaped))

        # summing log p over all variables (visible and hidden)
        sum_logs_probs = tf.reduce_sum(log_pdf_conditional_reshaped, axis=2) + tf.reduce_sum(log_pdf_prior_reshaped, axis=2) - tf.reduce_sum(log_pdf_latent_reshaped, axis=2)

        #batch_reshape = [self._model.n_z_samples, -1] + self._model._model_visible.batch_shape.as_list()[1:]
        #sum_logs_probs = tf.reshape(sum_logs_probs, batch_reshape)
        #logs_probs = tf.reduce_sum(sum_logs_probs, axis=2)
        logp_is = tf.reduce_logsumexp(sum_logs_probs, axis=0)

        return logp_is

    def create_loss(self):

        #TODO move up in AbstractAutoEncoder.
        # A TFDeepLearningModel model should be free to specify a model dependent loss..
        # compute the cost function, passing the model as a parameter
        self.loss, self.loss_nodes_to_log, self.loss_nodes_to_log_names, self.loss_nodes_to_log_filenames = self._cost_function(self)

    def encode(self, X, stp=False, sess = None, y=None):
        """Encode data by mapping it into the latent space."""

        sess = sess or self.get_raw_session()

        if stp:
            input = self.raw_x
        else:
            input = self.x

        feed_dict = {self.n_z_samples: 1, input: X} if y is None else {self.n_z_samples: 1, input: X, self.y: y}

        return sess.run([self.z,
                         self._approximate_posterior_params],
                        feed_dict=feed_dict)

    # could be merged with AE
    def decode(self, Z, sess = None, mask=None, y=None):
        """Decode latent vectors in input data."""
        sess = sess or self.get_raw_session()

        feed_dict = {self.z: Z} if y is None else {self.z: Z, self.y: y}

        image = sess.run(self.x_reconstruction_node,
                         feed_dict=feed_dict)

        if self.mask is None:
            return image
        else:
            return (mask * (image+1)/2)*2 - 1

    def generate(self, batch_size=1, sess = None):
        """ Generate data by sampling from latent space.

        """
        sess = sess or self.get_raw_session()

        # old implementation
        #prior_samples_np = sess.run(self._prior_samples, feed_dict = {self.n_z_samples : batch_size})

        #return sess.run(self.x_reconstruction_node, feed_dict = {self.z : prior_samples_np})

        # new faster implementation
        return sess.run(self.samples, feed_dict = {self.n_z_samples : batch_size})

    def reconstruct(self, X, stp=False, sess=None):
        """ Use VAE to reconstruct given data. """
        sess = sess or self.get_raw_session()

        if stp:
            input = self.raw_x
        else:
            input = self.x
            
        #if self.binary:
        return sess.run(self.x_reconstruction_node,
                        feed_dict={self.n_z_samples: 1, input: X})

    # used by VAELinearInterpolation
    def _compute_linear_interpolation(self, couples_of_images, n_images, session=None, mask=None):

        if session is None:
            session = self.get_raw_session()

        zs1, (means1, _) = self.encode([i[0] for i in couples_of_images], sess=session)
        zs2, (means2, _) = self.encode([i[1] for i in couples_of_images], sess=session)

        interpolations_means = [alpha*a + (1-alpha)*b for a, b in zip(means1, means2) for alpha in np.linspace(0,1,n_images + 2)]

        interpolations_zs = [alpha*np.array(z1) + (1-alpha)*np.array(z2) for z1, z2 in zip(zs1, zs2) for alpha in np.linspace(0,1,n_images + 2)]

        # create array of masks for all interpolations, if needed
        masks = np.repeat(mask, n_images + 2, axis=0) if mask is not None else None

        reconstructed_interpolations_means = self.decode(interpolations_means, session, mask=masks)
        reconstructed_interpolations_zs = self.decode(interpolations_zs, session, mask=masks)

        return reconstructed_interpolations_means, reconstructed_interpolations_zs

    # used by VAELinearInterpolation
    def _compute_fisher_rao_interpolation(self, couples_of_images, n_images, session=None, mask=None):

        if session is None:
            session = self.get_raw_session()

        _, (means1, covs1) = self.encode([i[0] for i in couples_of_images], sess=session)
        _, (means2, covs2) = self.encode([i[1] for i in couples_of_images], sess=session)


        interpolations_means = [alpha*a + (1-alpha)*b for a, b in zip(means1, means2) for alpha in np.linspace(0,1,n_images + 2)]


        interpolations_fisher_rao_means, interpolations_fisher_rao_covs  = self._gaussian_interpolations(means1, covs1, means2, covs2, n_images)

        interpolations_fisher_rao_means = interpolations_fisher_rao_means.reshape(-1,interpolations_fisher_rao_means.shape[-1])

        # same as for linear interpolations
        masks = np.repeat(mask, n_images + 2, axis=0) if mask is not None else None

        reconstructed_interpolations_fisher_rao = self.decode(interpolations_fisher_rao_means, session, mask=masks)

        return reconstructed_interpolations_fisher_rao

    def _gaussian_interpolations(self, mean_a, cov_a, mean_b, cov_b, steps):

        n_points = mean_a.shape[0]
        dim = mean_a.shape[1]
        s = steps+2

        # figure to debug
        #plt.figure()

        mean_t = np.zeros((n_points,s,dim))
        cov_t = np.zeros((n_points,s,dim))
        cov_t_e = np.zeros((n_points,s,dim))
        cov_t_m = np.zeros((n_points,s,dim))

        for j in range(n_points):
            for i in range(dim):

                if mean_a[j,i] < mean_b[j,i]:
                    mean1 = mean_b[j,i]
                    mean2 = mean_a[j,i]
                    cov1 = cov_b[j,i]
                    cov2 = cov_a[j,i]
                    flip = 1
                else:
                    mean1 = mean_a[j,i]
                    mean2 = mean_b[j,i]
                    cov1 = cov_a[j,i]
                    cov2 = cov_b[j,i]
                    flip = 0

                # lines through two points
                # https://bobobobo.wordpress.com/2008/01/07/solving-linear-equations-ax-by-c-0/
                a = np.sqrt(cov1) - np.sqrt(cov2)
                b = mean2 - mean1
                c = mean1*np.sqrt(cov2) - mean2*np.sqrt(cov1)
                mean = (mean1 + mean2)/2
                sqrt_cov = (np.sqrt(cov1) + np.sqrt(cov2))/2

                # line orthogonal to a line through a point
                # https://math.stackexchange.com/questions/646420/find-the-vector-equation-of-a-line-perpendicular-to-the-plane
                t = -sqrt_cov/b
                mean_0 = t*a + mean
                sqrt_cov_0 = 0

                r = np.sqrt(cov1+(mean_0-mean1)**2)
                r_bis = np.sqrt(cov2+(mean_0-mean2)**2)

                assert np.isclose(r,r_bis, atol=1e-05), "assert failed"

                # http://www.nabla.hr/PC-RectPolarCooSys4.htm
                theta1 = np.arctan(np.sqrt(cov1)/(mean1-mean_0))
                if (mean1-mean_0)<0:
                    theta1 += np.pi
                theta2 = np.arctan(np.sqrt(cov2)/(mean2-mean_0))
                if (mean2-mean_0)<0:
                    theta2 += np.pi

                for k in range(s):
                    t = theta2+k*(theta1-theta2)/(s-1)
                    mean_t[j,k,i] = r*np.cos(t) + mean_0
                    cov_t[j,k,i] = (r*np.sin(t))**2

                if flip:
                    mean_t[j,:,i] = np.flip(mean_t[j,:,i],axis=0)
                    cov_t[j,:,i] = np.flip(cov_t[j,:,i],axis=0)

                inv_cov1 = 1/cov1
                inv_cov2 = 1/cov2

                for k in range(s):
                    inv_cov = inv_cov2+k*(inv_cov1-inv_cov2)/(s-1)
                    cov_t_e[j,k,i] = 1/inv_cov

                for k in range(s):
                    cov = cov2+k*(cov1-cov2)/(s-1)
                    cov_t_m[j,k,i] = cov

                if flip:
                    cov_t_e[j,:,i] = np.flip(cov_t_e[j,:,i],axis=0)
                    cov_t_m[j,:,i] = np.flip(cov_t_m[j,:,i],axis=0)

            '''
            # plot debug
            plt.plot(mean_t[:,i],np.sqrt(cov_t[:,i]))
            plt.plot(mean_t[:,i],np.sqrt(cov_t_e[:,i]),'--')
            plt.plot(mean_t[:,i],np.sqrt(cov_t_m[:,i]),':')
            plt.plot(mean1,np.sqrt(cov1),'xr')
            plt.plot(mean2,np.sqrt(cov2),'xr')
            '''


        '''
        # plot debug
        plt.ylim(ymin=0)
        plt.axis('equal')
        plt.savefig("geodesic.png")

        pdb.set_trace()
        '''

        assert(np.allclose(mean_a,mean_t[:,-1],atol=0.00001)), pdb.set_trace()
        assert(np.allclose(mean_b,mean_t[:,0],atol=0.00001)), pdb.set_trace()
        assert(np.allclose(cov_a,cov_t[:,-1],atol=0.00001)), pdb.set_trace()
        assert(np.allclose(cov_b,cov_t[:,0],atol=0.00001)), pdb.set_trace()

        return mean_t, cov_t
