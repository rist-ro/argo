import matplotlib
matplotlib.use('Agg')

from argo.core.network.AbstractAutoEncoder import AbstractAutoEncoder

from argo.core.CostFunctions import CostFunctions
from .AENetwork import AENetwork

from argo.core.hooks.LoggingMeanTensorsHook import LoggingMeanTensorsHook
from .AEImagesReconstructHook import AEImagesReconstructHook
from .PCALatentVariablesHook import PCALatentVariablesHook
from .TwoDimPCALatentVariablesHook import TwoDimPCALatentVariablesHook

from datasets.Dataset import TRAIN_LOOP, TRAIN, VALIDATION


class AutoEncoder(AbstractAutoEncoder):
    """ Autoencoder (AE)

    TODO

    """

    launchable_name = "AE"

    default_params= {
        **AbstractAutoEncoder.default_params,

        #"warm_up_method" : None, # None | inverse_warm_up | warm_up
        #"minimal_covariance_x" : 0.1, # value added to the guassian diagonal entries of p(x|z)

        "cost_function": {"cost" : "l2"},

        # NB not implemented in the filename
        "epochs" : 2000,                     # number of epochs
    }

    def create_id(self):

        _id = self.launchable_name

        #_id += '-c' + CostFunctions.create_id(opts["cost_function"])
        _id += '-c' + self._cost_function.create_id(self._opts["cost_function"][1])

        super_id = super().create_id()
        network_id = self._network.create_id()

        _id += super_id + network_id

        return _id

    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):

        # notice that in the following opts is used, and not self._opts, until the
        # parent constructor is called
        
        #NB need to create the network before the super init because id generation depends on the network
        self._network = AENetwork(opts, "ae_network", seed=seed)
        self._cost_function = CostFunctions.instantiate_cost_function(opts["cost_function"], module_path= "vae")

        # self.covariance_parameterization = opts["covariance_parameterization"]

        # be careful with initialization in the following, since values have to be float

        '''
        if opts["cost_function"]["cost"]=="renyi" or opts["cost_function"]["cost"]=="irenyi":
            if "alpha" not in opts["cost_function"]:
                opts["cost_function"]["alpha"] = 1.0
            self.alpha_default = opts["cost_function"]["alpha"]

        # beta parameter for *elbo
        if opts["cost_function"]["cost"]=="elbo" or opts["cost_function"]["cost"]=="ielbo":
            if "beta" not in opts["cost_function"]:
                opts["cost_function"]["beta"] = 1.0
            self.beta_default = opts["cost_function"]["beta"]

        # h parameter for ielbo and irenyi
        if opts["cost_function"]["cost"]=="ielbo" or opts["cost_function"]["cost"]=="irenyi":
            if "h" not in opts["cost_function"]:
                opts["cost_function"]["h"] = 0.01
            self.h_default = opts["cost_function"]["h"]

        # normalize for ielbo and irenyi
        if opts["cost_function"]["cost"]=="ielbo" or opts["cost_function"]["cost"]=="irenyi":
            if "normalize" not in opts["cost_function"]:
                opts["cost_function"]["normalize"] = 0
            self.h_normalize = opts["cost_function"]["normalize"]
        '''

        # TODOfunctionlogger: change this name to avoid confusion with self.cost_function
        self._cost_function_tuple = opts["cost_function"]

        #TODO-ARGO2 why is samples never used in the code and where should it be used?
        #self.samples_number = opts["samples"]

        #TODO-ARGO2 this n_z_samples is a property of the network, is it really necessary to have it in the model?
        #TODO-ARGO2 how is this changing when we will use tf.distributions sampling and hopefully remove the replication?
        #TODO-ARGO2 n_z_samples should not be a placeholder but call Normal.samples(n_z_samples) produces the desired tensor, which can subsequently be reduced in the 0 axis to average over samples...
        #self.n_x_samples = self._network.n_x_samples

        # # replicate x
        # self.x_replicate = {}
        # self.x_target_replicate = {}

        # important nodes
        self._model_visible = None
        self.x_reconstruction_node = None

        super().__init__(opts, dirName, check_ops, gpu, seed)

    def create_hooks(self, config):
        hooks = super().create_hooks(config)

        #LOGGING HOOKS


        # check https://www.reddit.com/r/MachineLearning/comments/56m5o2/discussion_calculation_of_bitsdims/
        # to understand why I need to shid by log(256)


        # TODO the two shapes could be different, maybe use tf.shape(self.raw_x) (Riccardo)
        # dim_with_channels = np.prod(self.x_shape["train"])
        # dim_with_channels = np.prod(self.x_shape["eval"])
        #
        # # check https://www.reddit.com/r/MachineLearning/comments/56m5o2/discussion_calculation_of_bitsdims/
        # # to understand why I need to shid by log(256)
        # bits_dim = (self.loss/dim_with_channels - tf.log(256.0) )/tf.log(2.0)
        #

        tensors_to_average = [
         [[ self.loss ]
         ],
          self.loss_nodes_to_log
        ]
        tensors_to_average_names = [
         [[ "loss" ]
         ],
          self.loss_nodes_to_log_names
        ]
        tensors_to_average_plots = [
         [{"fileName" : "loss"}
         ],
          self.loss_nodes_to_log_filenames
        ]

        hooks.append(LoggingMeanTensorsHook(model = self,
                                            fileName = "log",
                                            dirName = self.dirName,
                                            tensors_to_average = tensors_to_average,
                                            tensors_to_average_names = tensors_to_average_names,
                                            tensors_to_average_plots = tensors_to_average_plots,
                                            average_steps = self._n_steps_stats,
                                            tensorboard_dir = self._tensorboard_dir,
                                            trigger_summaries = config["save_summaries"],
                                            plot_offset = self._plot_offset,
                                            train_loop_key = TRAIN_LOOP,
                                            datasets_keys = [VALIDATION],
                                            time_reference = self._time_reference_str
        )
                     )

        kwargs = config.get("ImagesReconstructHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(AEImagesReconstructHook(model = self,
                                                dirName = self.dirName,
                                                **kwargs)
                         )

        kwargs = config.get("TwoDimPCALatentVariablesHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(TwoDimPCALatentVariablesHook(model = self,
                                                dirName = self.dirName,
                                                tensors = [self.h],
                                                tensors_names = ['h'],
                                                datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
                                                **kwargs
                                               )
                         )

        kwargs = config.get("PCALatentVariablesHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(PCALatentVariablesHook(model = self,
                                                dirName = self.dirName,
                                                tensors = [self.h],
                                                tensors_names = ['h'],
                                                datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
                                                **kwargs
                                               )
                         )


        #TODO-ARGO2 append here specific AE loggers as hooks

        return hooks

    # def create_feedable_placeholders(self):
    #
    #     #TODO if needed put in right places. (__init__ of respective costfunctions)
    #     #TODO-ARGO2 this parameter handling is very confusing.. maybe a class for cost function should handle its own default parameters/nodes,
    #     #TODO-ARGO2 why do I need to handle them at this level?
    #     # if self._cost_function["cost"]=="renyi" or self._cost_function["cost"]=="irenyi":
    #     #     self.alpha = tf.placeholder_with_default(self.alpha_default, shape=(), name='alpha_renyi_bound')
    #     # if self._cost_function["cost"]=="elbo" or self._cost_function["cost"]=="ielbo":
    #     #     self.beta_coeff = tf.placeholder_with_default(self.beta_default, shape=(), name='beta_regularizer')
    #     # if self._cost_function["cost"]=="ielbo" or self._cost_function["cost"]=="irenyi":
    #     #     self.h_coeff = tf.placeholder_with_default(self.h_default, shape=(), name='h_coeff')
    #     #     self.h_normalize = tf.placeholder_with_default(self.h_normalize, shape=()), name='h_normalize')
    #     pass

    def create_network(self):
        # create autoencoder network
        self.h, self._model_visible = self._network(self.x, is_training=self.is_training)
        self.x_reconstruction_node = self._model_visible.reconstruction_node()

    def create_loss(self):
        self.loss, self.loss_nodes_to_log, self.loss_nodes_to_log_names, self.loss_nodes_to_log_filenames = self._cost_function(self)

    def encode(self, X, sess = None):
        """Encode data by mapping it into the latent space."""

        sess = sess or self.get_raw_session()

        return sess.run(self.h,
                        feed_dict={self.raw_x: X})

    # could be merged with VAE
    def decode(self, H, sess = None):
        """Decode latent vectors in input data."""
        sess = sess or self.get_raw_session()

        return sess.run(self.x_reconstruction_node,
                        feed_dict={self.h: H})

    def reconstruct(self, X, sess = None):
        """ Use AE to reconstruct given data. """
        sess = sess or self.get_raw_session()

        #if self.binary:
        return sess.run(self.x_reconstruction_node,
                        feed_dict={self.n_z_samples: 1, self.raw_x: X})


    def generate(self, batch_size=1, sess = None):
        pass



# CAN WE REMOVE ALL AFTER THIS LINE?




### REMOVE AFTER THIS LINE###
# DO NOT CROSS!
# TRESPASSERS WILL BE SHOT ON SIGHT...

    # # QUESTIONABLE utility.. methods of the Hook could be called instead..
#     def evaluate_loss(self, dataset_str):
#         raise Exception("it is not finished, and most probably it is not really needed.. see TODO here")
#
#         if dataset_str == "train":
#             raise Exception("The function 'evaluate_loss' should not be called over train")
#
#         session = self.get_raw_session()
#
#         # TODO CHECK IF NODE IS IN A COLLECTION I MADE APPOSITLY OR SOMEHOW THAT THE NODE DOES NOT ALREADY EXIST SOMEWHERE
#         # THE BEST WAY IS TO USE A COLLECTION SINCE A HOOK COULD HAVE CREATED IT!!
#         loss = self.loss
#         mean_values, mean_update_ops, mean_reset_ops = \
#             create_reset_metric(tf.metrics.mean, scope="mean_reset_metric/"+loss.name, values=loss)
#
#         # mean_values, mean_update_ops, mean_reset_ops can be anything, a single tf node, a list, a tuple or a dictionary of tf nodes
#         whatever_mean = evaluate_means_over_dataset(session,
#                                                     self.datasets_initializers[dataset_str],
#                                                     mean_values,
#                                                     mean_update_ops,
#                                                     mean_reset_ops)
#         return whatever_mean
#


        #
        # total_loss = 0.
        # n_steps = 0
        #
        # session.run(self.dataset_initializers)
        # while True:
        #     try:
        #         total_loss += session.run(self.loss)
        #     except tf.errors.OutOfRangeError:
        #         break
        #     n_steps += 1
        #
        # return total_loss / n_steps
        #
    '''
    # NOT USED?
    def test_cost(self, X, X_original=None):
        # I added to add here the number of samples
        if X_original is None:
            return self.sess.run(self.cost_function.log_p_x_z, feed_dict={self.n_z_samples : self.samples,
                                                                          self.x : X})
        else:
            return self.sess.run(self.cost_function.log_p_x_z, feed_dict={self.n_z_samples : self.samples,
                                                                          self.x_tilde: X,
                                                                          self.x : X_original})

    '''

    #
    # # This redefinition of the method is required, since after the creation of the object
    # # GaussianVariationalAutoEncoder I have a graph which needs to be passed to
    # # VAELogpImportanceSamplingLogger, to create the importance sampling graph
    # def register_vae_log_p_importance_sampling_logger(self, logger):
    #     super(GaussianVariationalAutoEncoder, self).register_vae_log_p_importance_sampling_logger(logger)
    #
    #     with self.sess.graph.as_default():
    #         # Create the graph for the importance sampling logger
    #         if self.binary:
    #             # TODO: this should be removed, and I would pass the same object independentely from binary
    #             reshaped_x_reconstr_mean_samples = tf.reshape(self.x_reconstr_mean_samples, [-1, self.input_size])
    #         else:
    #             reshaped_x_reconstr_mean_samples = None
    #
    #         x_imp_samples = tf.reshape(tf.tile(self.x, [1, self.n_z_samples]), [-1, self.input_size])
    #
    #         # TODO this should be removed from here, and object should be passed which implements
    #         # this behavior, according to the output model
    #         if self.binary == 1:
    #             log_pdf_conditional = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_imp_samples,
    #                                                                         logits=x_reconstr_imp_samples)
    #             log_pdf_conditional = tf.reduce_sum(log_pdf_conditional, 1)
    #
    #         else:
    #             # NB here I sample only one point, due to a difference in the way I compute marginal log p
    #             #if self.synthetic:
    #             #    log_pdf_conditional = self._model_visible["train"].log_pdf(x_imp_samples)
    #             #else:
    #             log_pdf_conditional = self._model_visible["train"].log_pdf(x_imp_samples)
    #
    #         with tf.variable_scope('importance_sampling'):
    #             logger.create_importance_sampling_graph(self.synthetic, self.x, self.n_z_samples, self.z, self.log_pdf_z, log_pdf_conditional, reshaped_x_reconstr_mean_samples, self._model_visible["train"])
    #
    # # This redefinition of the method is required, since after the creation of the object
    # # GaussianVariationalAutoEncoder I need to add notes to the graph based on the function
    # # which hae to be estimated
    # def register_vae_estimate_function_logger(self, logger):
    #     super(GaussianVariationalAutoEncoder, self).register_vae_estimate_function_logger(logger)
    #
    #     # Create the graph for the estimation of a function based on sample, such as Renyi bounds
    #     if self.binary:
    #         # TODO: this should be removed, and I would pass the same object independentely from binary
    #         reshaped_x_reconstr_mean_samples = tf.reshape(self.x_reconstr_mean_samples, [-1, self.input_size])
    #     else:
    #         reshaped_x_reconstr_mean_samples = None
    #
    #     # TODOfunctionestimate: we need to remove the objects passed as arguments in case some methods
    #     # which create graphs are called, otherwise graph is replicaed. For instance, most likely
    #     # we should not need to pass self._gaussian_model_latent["train"]
    #     with self.sess.graph.as_default():
    #         logger.create_estimate_function_graph(self.binary, self.synthetic, self.x, self.x_replicate["train"], self.n_z_samples, self.z, self.log_pdfs_latent, self.x_reconstr_mean_samples, reshaped_x_reconstr_mean_samples, self._gaussian_model_latent["train"], self._model_visible["train"], self.alpha, self.labmda_coeff)
    #
    #
    # def old_train(self, ds, opts):
    #     """
    #     Warm up: gradually introduce the KL divergence in the cost function, as suggested in Ladder VAE paper
    #     warm_up_method = {-1, 0, 1}: 0 = no warm up
    #                                 -1 = inverse warm up (scale of KL decreases until it becomes 1)
    #                                  1 = warm up (scale of KL increases until it becomes 1)
    #     """
    #
    #     stochastic = opts["stochastic"] if "stochastic" in opts else 0
    #     stochastic_noise_param = opts["stochastic_noise_param"] if "stochastic_noise_param" in opts else 0
    #     training_epochs = opts["epochs"] if "epochs" in opts else 10
    #     warm_up_method = opts["warm_up_method"] if "warm_up_method" in opts else 1
    #     batch_size = opts["batch_size"] if "batch_size" in opts else 128
    #     batch_size_test = opts["batch_size_test"] if "batch_size_test" in opts else 128
    #
    #     # noise parameters
    #     # for latent_noise see the constructor
    #     bit_flip = opts["bit_flip"] if "bit_flip" in opts else 0
    #     drop_out = opts["drop_out"] if "drop_out" in opts else 0
    #
    #     n_samples_train = ds.n_samples_train
    #     n_samples_test = ds.n_samples_test
    #
    #     perturbed_train_set = getattr(ds, "perturbed_train_set_x", None)
    #     perturbed_test_set = getattr(ds, "perturbed_test_set_x", None)
    #     assert((perturbed_train_set is not None and perturbed_test_set is not None) or (perturbed_train_set is None and perturbed_test_set is None), "Either both train and test perturbed are None, or both train and perturbed must be specific in the dataset")
    #     if (perturbed_train_set is not None and perturbed_test_set is not None):
    #         perturbed_dataset = 1
    #     else:
    #         perturbed_dataset = 0
    #
    #     if perturbed_dataset:
    #         train_set = ds.perturbed_train_set_x
    #         test_set = ds.perturbed_test_set_x
    #     else:
    #         train_set = ds.train_set_x
    #         test_set = ds.test_set_x
    #
    #     #binary = dataset["binarized"]
    #
    #     # TO BE REMOVED, use ds.x_shape instead
    #     self._image_width, self._image_height, n_channels = ds.x_shape
    #
    #     # This may not be general enough, in case of non images
    #     if self._vae_latent_vars_classification_logger:
    #         self._vae_latent_vars_classification_logger.set_image_shape((self._image_height, self._image_width))
    #         self._vae_latent_vars_classification_logger.set_color(ds.color_images)
    #
    #     total_batch_train = int(n_samples_train / batch_size)
    #     total_batch_test = int(n_samples_test / batch_size_test)
    #     n_samples_train = total_batch_train * batch_size
    #     n_samples_test = total_batch_test * batch_size_test
    #     warm_up = 1
    #
    #     # Loop over epochs
    #     if self._images_generator:
    #         # now that I know the size of the images I can give this information to the image generator
    #         self._images_generator.set_color(ds.color_images)
    #         self._images_generator.image_size = (self._image_height, self._image_width)
    #         n_images = self._images_generator.number_images_columns*self._images_generator.number_images_rows
    #
    #         # TODO this should be changed in case of multiple stochastic layers
    #         dim_latent_space = self.network_architecture["stochastic_latent"][0]["size"]
    #         z_generate_from_prior = np.random.normal(size=(n_images, dim_latent_space))
    #
    #     if self._images_regenerator:
    #         # now that I know the size of the images I can give this information to the image regenerator
    #         self._images_regenerator.set_color(ds.color_images)
    #         self._images_regenerator.image_size = (self._image_height, self._image_width)
    #
    #
    #     if self.synthetic:
    #         true_log_likelihood_train = ds["gaussian_class"].compute_log_likelihood(ds.train_set_x)
    #         print("Log Likelihood train: %.3f" %true_log_likelihood_train)
    #         true_log_likelihood_test = ds["gaussian_class"].compute_log_likelihood(ds.test_set_x)
    #         print("Log Likelihood test: %.3f" %true_log_likelihood_test)
    #
    #     self.set_summaries()
    #
    #     #I need to put this fake run step just to activate the should stop, otherwise if model is trained it would do an extra step (maybe bug?)
    #     self.sess.run(self.global_step)
    #     #TODO-ARGO2 temporary solution for gradual change, this is not the orthodox way
    #     if self.sess.should_stop():
    #         print("training not needed stopping criteria already met.")
    #         return
    #
    #     for epoch in range(1,training_epochs+1):
    #         #TODO-ARGO2 temporary solution for gradual change, this is not the orthodox way
    #         if self.sess.should_stop():
    #             break
    #
    #         perm = np.random.permutation(n_samples_train)
    #         warm_up = 1
    #         # if epoch < 0:
    #         if warm_up_method==1:
    #             """Increase the KL scale from 0 to 1, over 200 epochs
    #             """
    #             warm_up = float((epoch - 1)/50)
    #         if warm_up_method==-1:
    #             """Decrease the KL scale from 10 (epoch 0) to 1 (epoch 200)
    #             In general the formula is n - ((n-1)*epoch)/200
    #             """
    #             warm_up = 10 - float(9*epoch/50)
    #
    #         # start timer for TF
    #         start_tf = timeit.default_timer()
    #
    #         avg_cost_train = 0.
    #         avg_cost_test = 0.
    #         avg_latent_loss_train = 0.
    #         avg_latent_loss_test = 0.
    #         grad_norm_mean = []
    #         weights_norm_mean = []
    #
    #         if not isinstance(train_set, np.ndarray):
    #             # this is the case of not ndarray (such as continuous mnist)
    #             raise Exception("not implemented yet, why?")
    #             pass
    #         else:
    #             # stochastic = 1 means sampling from the data using Bernoulli distribution after
    #             # every epoch for binarized MNIST; this prevents overfitting, see LVAE and IWAE papers
    #
    #             # TODO clipping shold be after the following if, so that I clip even if I don't
    #             # perturb the dataset
    #
    #
    #             if stochastic:
    #                 if self.binary:
    #                     augmented_train_set = utils.sample_discrete_from_continuous(train_set)
    #                     augmented_test_set = utils.sample_discrete_from_continuous(test_set)
    #                 else:
    #                     augmented_train_set, noise_train = utils.add_gaussian_noise_and_clip(train_set,self.stochastic_noise_param, low=0, high=1)
    #                     augmented_test_set, noise_test = utils.add_gaussian_noise_and_clip(test_set,self.stochastic_noise_param, low=0, high=1)
    #                 processed_train_set = augmented_train_set
    #                 processed_test_set = augmented_test_set
    #
    #             else:
    #                 processed_train_set = train_set
    #                 processed_test_set = test_set
    #
    #         # this is the case of ndarray (such as binarized mnist and cifar)
    #         # manualy performing the permutation
    #         permuted_processed_train_set = processed_train_set[perm]
    #         #permuted_train_set = train_set[perm]
    #
    #         if self.binary:
    #             if perturbed_dataset:
    #                 raise Exception("Not implemented for the moment")
    #             else:
    #                 permuted_target_train_set = ds.train_set_x[perm]
    #                 target_test_set = ds.test_set_x
    #                 if not self.denoising and stochastic:
    #                     # use the same noise for continuous datasets
    #                     permuted_target_train_set = permuted_processed_train_set
    #                     target_test_set = processed_test_set
    #         else:
    #             permuted_target_train_set = ds.train_set_x[perm]
    #             target_test_set = ds.test_set_x
    #             if not self.denoising and stochastic:
    #                 # use the same noise for continuous datasets
    #                 permuted_target_train_set = utils.clip(ds.train_set_x[perm] + noise_train[perm])
    #                 target_test_set = utils.clip(ds.test_set_x + noise_test)
    #
    #
    #         # Loop over batch
    #         for i in range(0, n_samples_train, batch_size):
    #             #TODO-ARGO2 temporary solution for gradual change, this is not the orthodox way
    #             if self.sess.should_stop():
    #                 break
    #
    #             perm_batch = perm[i:i + batch_size]
    #
    #             '''
    #             if not isinstance(train_set, np.ndarray):
    #                 # not ndarray (such as continuous mnist)
    #                 # TODO control shuffle
    #                 # TODO remove this
    #                 raise Exception("not implemented yet, why?")
    #                 X, _ = train_set.next_batch(batch_size)
    #             else:
    #             '''
    #             # ndarray (such as binarized mnist and cifar)
    #             if i + batch_size > permuted_processed_train_set.shape[0] : break
    #             X = permuted_processed_train_set[i:i + batch_size]
    #             X_target = permuted_target_train_set[i:i + batch_size]
    #
    #             # Add random bit-flip:
    #             if bit_flip>0:
    #                 flip = np.random.binomial(1, bit_flip, size=X.shape)
    #                 X_target = (X_target + flip) % 2
    #                 X = (X + flip) % 2
    #             # Add drop-out noise:
    #             if drop_out>0:
    #                 drop = np.random.binomial(1, 1 - drop_out, size=X.shape)
    #                 X = X * drop
    #                 X_target = X_target * drop
    #
    #             # I need to shring to make sure that I don't have 0 and 1 for continuous
    #             # data, otherwise I may have "division by 0" for the likelihood of the logit-normal00
    #             if self.binary==0 and self.synthetic==0:
    #                 # rescale set in [0,1]
    #                 X = utils.rescale(X, self.rescale)
    #                 X_target = utils.rescale(X_target, self.rescale)
    #
    #             # Fit training using batch data
    #
    #             # TODO ALL (or at least part) of THIS train FUNCTION MUST BE MOVED UP IN TFDeepLearningAlgorithm
    #
    #             # TODO maybe returned args by partial_fit should be a dictionary, otherwise is very confusing.
    #             if self.denoising:
    #                 args, summaries_lists = self.partial_fit(X,
    #                                         X_target = X_target,
    #                                         warm_up = warm_up,
    #                                         log_gradient_current_epoch=self._gradient_logger and self._gradient_logger.log_epoch(epoch))
    #             else:
    #                 args, summaries_lists = self.partial_fit(X,
    #                                         warm_up = warm_up,
    #                                         log_gradient_current_epoch=self._gradient_logger and self._gradient_logger.log_epoch(epoch))
    #
    #             #TODO XXX
    #             # TODO YYYY
    #             #if i==0:
    #             #    pdb.set_trace()
    #
    #             #log_p_x_z = self.test_cost(X)
    #
    #             #TODO-ARGO2 retrieving global_step to pass it to the summaries (will not be needed when using hooks)
    #             gstep = args[0]
    #             opt = args[1]
    #             cost = args[2]
    #             regularization = args[3]
    #             latent_loss = args[4]
    #
    #             #this is to have summaries that write on a separate files
    #             for ks in summaries_lists:
    #                 writers = self.summary_writers[ks]
    #                 summaries = summaries_lists[ks]
    #                 for wr,summ in zip(writers, summaries):
    #                     wr.add_summary(summ, global_step=gstep)
    #
    #
    #             '''
    #             # this should happen independently, just a check for non implemented loggers
    #             if 0 in [0, -2, 3]:
    #                 z_mean = args[3]
    #                 z_covariance_parameterization = args[4]
    #                 x_mean = args[5]
    #                 if self.binary == 1:
    #                     index = 6
    #                 else:
    #                     x_covariance_parameterization = args[6]
    #                     index = 7
    #             else:
    #                 index = 3
    #             '''
    #             # TODO variable index has never been defined :o
    #             if self._gradient_logger and self._gradient_logger.log_epoch(epoch):
    #                 if 0==1: #self.k TODO IF 0==1, Really???
    #                     index = index + 1
    #                 # pdb.set_trace()
    #                 grads_and_vars = args[index]
    #                 index = index + 1
    #             ''' TODOTF to be deleted
    #             if self._log_tf==1:
    #                 summary = args[index]
    #                 index = index + 1
    #             '''
    #             # not used
    #             if self._check_ops==1:
    #                 numerics_ops = args[index]
    #                 index = index + 1
    #             ''' TODOTF to be deleted
    #             if self._log_tf==1:
    #                 self._summary_writer.add_summary(summary,epoch * total_batch_train + i/batch_size)
    #             '''
    #             if self._gradient_logger and self._gradient_logger.log_epoch(epoch):
    #                 self._gradient_logger.log(epoch, (i/batch_size), grads_and_vars)
    #
    #             # compute average loss
    #             avg_cost_train += cost * batch_size
    #             avg_latent_loss_train += latent_loss * batch_size
    #
    #             # importance sampling estimation of log p
    #             if self._vae_log_p_importance_sampling_logger and self._vae_log_p_importance_sampling_logger.log_epoch(epoch):
    #                 marg_train = self._vae_log_p_importance_sampling_logger.compute_log_p_batch(self.x, X, self.n_z_samples, self.sess)
    #                 self._vae_log_p_importance_sampling_logger.log_train(marg_train / n_samples_train)
    #
    #             # estimate function logger
    #             if self._vae_estimate_function_logger and self._vae_estimate_function_logger.log_epoch(epoch):
    #                 function_train = self._vae_estimate_function_logger.compute_estimate_function_batch(self.x, X, self.n_z_samples, self.sess, self.alpha, self.beta_coeff)
    #                 #pdb.set_trace()
    #                 self._vae_estimate_function_logger.log_train(function_train, n_samples_train)
    #
    #         #TODO-ARGO2 what to fix down here?
    #         # FIX THIS
    #         if self._gradient_logger and self._gradient_logger.log_epoch(epoch):
    #             self._gradient_logger.plot()
    #
    #         # compute average loss
    #         avg_cost_train = - avg_cost_train / n_samples_train
    #         avg_latent_loss_train = avg_latent_loss_train / n_samples_train
    #
    #         for i in range(0, n_samples_test, batch_size_test):
    #             #TODO-ARGO2 temporary solution for gradual change, this is not the orthodox way
    #             if self.sess.should_stop():
    #                 break
    #
    #             '''
    #             if not isinstance(train_set, np.ndarray):
    #                 raise Exception("not implemented yet, why?")
    #                 X, _ = perturbed_test_set.next_batch(batch_size_test)
    #             else:
    #             '''
    #             if i + batch_size_test > processed_test_set.shape[0] : break
    #             X = processed_test_set[i:i + batch_size]
    #             X_target = target_test_set[i:i + batch_size_test]
    #
    #             if self.binary==0 and self.synthetic==0:
    #                 X_target = utils.rescale(X_target, self.rescale)
    #                 X = utils.rescale(X, self.rescale)
    #
    #
    #             # Fit test using batch data
    #             if self.denoising:
    #                 cost, latent_loss = self.test_fit(X, X_target = X_target)
    #             else:
    #                 cost, latent_loss = self.test_fit(X)
    #             # Compute average loss
    #             avg_cost_test += cost * batch_size_test
    #             avg_latent_loss_test += latent_loss * batch_size_test
    #
    #             # importance sampling estimation of log p
    #             if self._vae_log_p_importance_sampling_logger and self._vae_log_p_importance_sampling_logger.log_epoch(epoch):
    #                 marg_test = self._vae_log_p_importance_sampling_logger.compute_log_p_batch(self.x, X, self.n_z_samples, self.sess)
    #                 self._vae_log_p_importance_sampling_logger.log_test(marg_test / n_samples_test)
    #
    #             # estimate function logger
    #             if self._vae_estimate_function_logger and self._vae_estimate_function_logger.log_epoch(epoch):
    #                 function_test = self._vae_estimate_function_logger.compute_estimate_function_batch(self.x, X, self.n_z_samples, self.sess, self.alpha, self.beta_coeff)
    #                 self._vae_estimate_function_logger.log_test(function_test, n_samples_test)
    #
    #
    #         # importance sampling estimation of log p
    #         if self._vae_log_p_importance_sampling_logger and self._vae_log_p_importance_sampling_logger.log_epoch(epoch):
    #             self._vae_log_p_importance_sampling_logger.log(epoch)
    #             self._vae_log_p_importance_sampling_logger.plot()
    #
    #         # estimate function
    #         if self._vae_estimate_function_logger and self._vae_estimate_function_logger.log_epoch(epoch):
    #             self._vae_estimate_function_logger.log(epoch)
    #             self._vae_estimate_function_logger.plot()
    #
    #
    #         avg_cost_test = - avg_cost_test / n_samples_test
    #         avg_latent_loss_test = avg_latent_loss_test / n_samples_test
    #
    #         # end timer for TF
    #         stop_tf = timeit.default_timer()
    #         time = stop_tf - start_tf
    #
    #         graph_size = self.graph_size
    #         default_graph_size = len([n.name for n in tf.get_default_graph().as_graph_def().node])
    #
    #         # start timer for loggers (some loggers, not all of them)
    #         start_log = timeit.default_timer()
    #
    #         if self._vae_latent_vars_model_logger and self._vae_latent_vars_model_logger.log_epoch(epoch):
    #
    #             selected_points = self._vae_latent_vars_model_logger.get_selected_points("train")
    #             X = perturbed_train_set[selected_points]
    #             z_mean, z_covariance_parameters, z = self.encode(X)
    #             for j, p in enumerate(selected_points):
    #                 self._vae_latent_vars_model_logger.log(p,
    #                                                        "train",
    #                                                        epoch,
    #                                                        z_mean[j],
    #                                                        self._gaussian_model_latent["train"].get_covariance_from_parameters_np(z_covariance_parameters[j]))
    #                 self._vae_latent_vars_model_logger.plot(p,"train")
    #
    #
    #             selected_points = self._vae_latent_vars_model_logger.get_selected_points("test")
    #             X = perturbed_test_set[selected_points]
    #             z_mean, z_covariance_parameters, z = self.encode(X)
    #             for j, p in enumerate(selected_points):
    #                 self._vae_latent_vars_model_logger.log(p,
    #                                                        "test",
    #                                                        epoch,
    #                                                        z_mean[j],
    #                                                        self._gaussian_model_latent["train"].get_covariance_from_parameters_np(z_covariance_parameters[j]))
    #                 self._vae_latent_vars_model_logger.plot(p,"test")
    #
    #
    #         if self._vae_visible_vars_model_logger and self._vae_visible_vars_model_logger.log_epoch(epoch):
    #
    #             selected_points = self._vae_visible_vars_model_logger.get_selected_points("train")
    #             X = perturbed_train_set[selected_points]
    #             z_mean, z_covariance_parameters, z = self.encode(X)
    #             x_mean, x_covariance_parameters = self.generate(z)
    #             for j, p in enumerate(selected_points):
    #                 self._vae_visible_vars_model_logger.log(p,
    #                                                        "train",
    #                                                        epoch,
    #                                                        x_mean[j],
    #                                                        self._model_visible["train"].get_covariance_from_parameters_np(x_covariance_parameters[j]))
    #                 self._vae_visible_vars_model_logger.plot(p,"train")
    #
    #
    #             selected_points = self._vae_visible_vars_model_logger.get_selected_points("test")
    #             X = perturbed_test_set[selected_points]
    #             z_mean, z_covariance_parameters, z = self.encode(X)
    #             x_mean, x_covariance_parameters = self.generate(z)
    #             for j, p in enumerate(selected_points):
    #                 self._vae_visible_vars_model_logger.log(p,
    #                                                        "test",
    #                                                        epoch,
    #                                                        x_mean[j],
    #                                                        self._model_visible["train"].get_covariance_from_parameters_np(x_covariance_parameters[j]))
    #                 self._vae_visible_vars_model_logger.plot(p,"test")
    #
    #
    #
    #         '''
    #         if self._vae_repeated_reconstructions_logger and self._vae_repeated_reconstructions_logger.log_epoch(epoch):
    #
    #             selected_points = self._vae_repeated_reconstructions_logger.get_selected_points("train")
    #             X = perturbed_train_set[selected_points]
    #             for iteration in range(self._vae_repeated_reconstructions_logger.n_reconstructions):
    #                 X, z_mean, z, z_cov_params, cov_params_x = self.reconstruct(X)
    #             for j, p in enumerate(selected_points):
    #
    #
    #                 self._vae_visible_vars_model_logger.log(p,
    #                                                        "train",
    #                                                        epoch,
    #                                                        x_mean[j],
    #                                                        self._model_visible["train"].get_covariance_from_parameters(x_covariance_parameters[j]))
    #                 self._vae_visible_vars_model_logger.plot(p,"train")
    #
    #
    #             selected_points = self._vae_visible_vars_model_logger.get_selected_points("test")
    #             X = perturbed_test_set[selected_points]
    #             z_mean, z_covariance_parameters, z = self.encode(X)
    #             x_mean, x_covariance_parameters = self.generate(z)
    #             for j, p in enumerate(selected_points):
    #                 self._vae_visible_vars_model_logger.log(p,
    #                                                        "test",
    #                                                        epoch,
    #                                                        x_mean[j],
    #                                                        self._model_visible["train"].get_covariance_from_parameters(x_covariance_parameters[j]))
    #                 self._vae_visible_vars_model_logger.plot(p,"test")
    #         '''
    #
    #         if self._images_generator and self._images_generator.log_epoch(epoch):
    #             images, _ = self.generate(z_generate_from_prior)
    #             #resized_images = [image.reshape(self._image_height, self._image_width) for image in images]
    #             self._images_generator.save_images(images, epoch)
    #
    #         if self._images_regenerator and self._images_regenerator.log_epoch(epoch):
    #
    #             def regenerate_image(dataset, suffix):
    #                 width = self._images_regenerator.number_images_columns
    #                 height = self._images_regenerator.number_images_rows
    #                 X = dataset[:width*height]
    #                 means, cov, z = self.encode(X)
    #                 images_mu, _ = self.generate(means)
    #                 images_z, _ = self.generate(z)
    #
    #                 height = height*3
    #
    #                 composite = np.zeros((X.shape[0]*3,X.shape[1]))
    #                 for i in range(0,int(height),3):
    #                     composite[int(i*width):int((i+1)*width)] = X[int(i/3*width):int((i/3+1)*width)]
    #                     composite[int((i+1)*width):int((i+2)*width)] = images_mu[int(i/3*width):int((i/3+1)*width)]
    #                     composite[int((i+2)*width):int((i+3)*width)] = images_z[int(i/3*width):int((i/3+1)*width)]
    #
    #                 self._images_regenerator.save_images(composite, epoch, width=width, height=height, fileNameSuffix = suffix, title="1st line: orig image; 2nd line: recostr mean; 3rd line: reconstr z ")
    #
    #             regenerate_image(perturbed_train_set,"perturbed_train")
    #             regenerate_image(train_set,"train")
    #             regenerate_image(perturbed_test_set,"perturbed_test")
    #             regenerate_image(test_set,"test")
    #
    #         # check if the stats logger is enabled (e.g. pca_logger or corr_logger),
    #         # in case the andwer is yes, there are some common precomputation to be done based on encode()
    #         if (self._vae_latent_vars_pca_logger and self._vae_latent_vars_pca_logger.log_epoch(epoch)) or (self._vae_latent_vars_corr_logger and self._vae_latent_vars_corr_logger.log_epoch(epoch)) or (self._vae_latent_vars_classification_logger and self._vae_latent_vars_classification_logger.log_epoch(epoch)):
    #
    #             means_train, cov_train, samples_train = self.encode(train_set[perm])
    #             means_test, cov_test, samples_test = self.encode(perturbed_test_set)
    #             means = (means_train, means_test)
    #             samples = (samples_train, samples_test)
    #
    #             #samples_train = samples_train[::self.samples,:]
    #             #samples_test = samples_test[::self.samples,:]
    #
    #             # Plot the PCA eigenvalues of latent means and samples if the flag is set
    #             if self._vae_latent_vars_pca_logger and self._vae_latent_vars_pca_logger.log_epoch(epoch):
    #                 self._vae_latent_vars_pca_logger.plot_pca_eigenvalues(means, samples, epoch)
    #
    #             # Plot the correlations plots of latent means and samples if the flag is set
    #             if self._vae_latent_vars_corr_logger and self._vae_latent_vars_corr_logger.log_epoch(epoch):
    #                 self._vae_latent_vars_corr_logger.corr_heatmap(means, samples, epoch)
    #                 self._vae_latent_vars_corr_logger.corr_pairwise(means[0], epoch)
    #
    #             if self._vae_latent_vars_classification_logger and self._vae_latent_vars_classification_logger.log_epoch(epoch):
    #                 labels_train = ds.train_set_y[perm]
    #                 #pdb.set_trace()
    #                 labels_test = ds.test_set_y
    #                 self._vae_latent_vars_classification_logger.log(train_set[perm], perturbed_test_set, means_train, cov_train, samples_train, labels_train, means_test, cov_test, samples_test, labels_test, epoch)
    #                 self._vae_latent_vars_classification_logger.plot(epoch)
    #
    #         # Plot the generated images obtained from interpolating between two latent means or samples (linear interpolation)
    #         if self._vae_latent_lin_interpolate and self._vae_latent_lin_interpolate.log_epoch(epoch):
    #
    #             for c in self._vae_latent_lin_interpolate.images_couples:
    #                 x1 = train_set[c[0]].reshape(1,-1)
    #                 x2 = train_set[c[1]].reshape(1,-1)
    #
    #                 mean1, cov1, sample1 = self.encode(x1)
    #                 mean2, cov2, sample2 = self.encode(x2)
    #
    #                 means_t = self._vae_latent_lin_interpolate.linear_interpolation(mean1, mean2)
    #                 samples_t = self._vae_latent_lin_interpolate.linear_interpolation(sample1[0], sample2[0])
    #
    #                 # pdb.set_trace()
    #                 # TODO THIS IS WRONG!!
    #                 cov1_diag = np.exp(cov1[0])
    #                 cov2_diag = np.exp(cov2[0])
    #
    #                 means_geo_t, covs_geo_t = self._vae_latent_lin_interpolate.geo_interpolation(mean1[0], cov1_diag, mean2[0], cov2_diag)
    #
    #                 samples_geo_t = np.random.normal(means_geo_t,covs_geo_t)
    #
    #                 steps = len(self._vae_latent_lin_interpolate._steps)
    #
    #                 # NB: samples are the probabilities of the samples
    #                 generated_from_means, _ = [sp.special.expit(self.generate(means_t[i].reshape(1,-1)).reshape(self._image_height, self._image_width)) for i in range(steps)]
    #                 generated_from_samples, _ = [sp.special.expit(self.generate(samples_t[i].reshape(1,-1)).reshape(self._image_height, self._image_width)) for i in range(steps)]
    #                 generated_from_geodesic, _ = [sp.special.expit(self.generate(samples_geo_t[i].reshape(1,-1)).reshape(self._image_height, self._image_width)) for i in range(steps)]
    #                 generated_from_means_geodesic, _ = [sp.special.expit(self.generate(means_geo_t[i].reshape(1,-1)).reshape(self._image_height, self._image_width)) for i in range(steps)]
    #
    #                 self._vae_latent_lin_interpolate.interpolated_images(c[0], c[1], generated_from_means, generated_from_samples, generated_from_geodesic, generated_from_means_geodesic, epoch)
    #
    #         # NO NEED ANYMORE, saving with Hooks
    #         # # Save model
    #         # if self._model_saver:
    #         #     self._model_saver.save_model(self.sess, epoch)
    #
    #         # end timer for TF
    #         stop_log = timeit.default_timer()
    #         time_loggers = stop_log - start_log
    #
    #         # Final logger for VAE
    #         if self._vae_logger and self._vae_logger.log_epoch(epoch):
    #             self._vae_logger.log(epoch, avg_cost_train, avg_latent_loss_train, avg_cost_test, avg_latent_loss_test, graph_size, default_graph_size, time, time_loggers)
    #         if self._vae_logger and self._vae_logger.plot_epoch(epoch):
    #             self._vae_logger.plot()
    #
    #     # NO NEED ANYMORE, saving with Hooks
    #     # print("ready to save")
    #     # if self._model_saver:
    #     #     self._model_saver.save_model(self.sess, epoch, force_save = True)
    #     print("saving completed")
    #

    '''
    # NOT USED?
    def test_cost(self, X, X_original=None):
        # I added to add here the number of samples
        if X_original is None:
            return self.sess.run(self.cost_function.log_p_x_z, feed_dict={self.n_z_samples : self.samples,
                                                                          self.x : X})
        else:
            return self.sess.run(self.cost_function.log_p_x_z, feed_dict={self.n_z_samples : self.samples,
                                                                          self.x_tilde: X,
                                                                          self.x : X_original})

    '''

    # def partial_fit(self, X, X_target=None, warm_up=1, log_gradient_current_epoch=False):
    #     """Train model based on mini-batch of input data.
    #     Return cost of mini-batch.
    #     """
    #
    #     ks = [-2, 0, 3, 1]
    #     if 0 in ks:
    #         # TODO XXX
    #         # based on the choice of the "learning rule", I could evaluate either:
    #         # self.optimizer
    #         # or
    #         # self.optimizer_encoder and self.optimizer_deconder
    #
    #         # TODO XXX
    #         # we may choose to return the values for the cost and the latent_loss, one after
    #         # the first optimization step, and the other after the second.
    #         # notice that we prefer not to call sessio.run twice, to avoid delays in moving data
    #         # between RAM and GPU
    #
    #         # TODO XXX
    #         # return two values for cost and latent_loss implies to change the loggers, since I have
    #         # to log on file multiple values
    #
    #         # I may need to evaluate the mean and covariances if I want to log information in
    #         # the main algorithm
    #         # TODO Luigi: check if this could be optimized somehow
    #         if self.binary == 0:
    #             arg = (self.optimizer, self.cost, self.regularizer, self.latent_loss)
    #         else:
    #             arg = (self.optimizer, self.cost, self.regularizer, self.latent_loss)
    #     else:
    #         arg = (self.optimizer, self.cost, self.regularizer, self.latent_loss)
    #
    #     #TODO what is this "if frenzy" up here? all cases are the same ... !
    #
    #     arg = (self.global_step, *arg)
    #
    #     if self._gradient_logger and log_gradient_current_epoch:
    #         arg = arg + (self.grads_and_vars,)
    #     ''' TODOTF to be deleted
    #     if self._log_tf==1:
    #         arg = arg + (self._merged_summary_op,)
    #     '''
    #
    #     # check_ops is used to evalute extra nodes added for deubg porposes
    #     if self._check_ops==1:
    #         arg = arg + (self._numerics_ops,)
    #
    #     # TODO XXX
    #     # test to be removed, made by Luigi and Sabin to see the gradient of part of the cost
    #     #arg = arg + (self.my_grads_and_vars,)
    #
    #     if X_target is None:
    #         return self.sess.run((arg, self.summary_nodes), feed_dict={self.x: X, self.n_z_samples: self.samples, self._warm_up: warm_up})
    #     else:
    #         return self.sess.run((arg, self.summary_nodes), feed_dict={self.x: X,
    #                                              self.x_target: X_target,
    #                                              self.n_z_samples: self.samples,
    #                                              self._warm_up: warm_up})
    #
    # def test_fit(self, X, X_target=None, warm_up=1):
    #     """Test model over input data.
    #     Return cost of mini-batch.
    #     """
    #     if X_target is None:
    #         cost, latent_loss = self.sess.run((self.cost, self.latent_loss),
    #                                           feed_dict={self.n_z_samples: self.samples,
    #                                                      self.x: X,
    #                                                      self._warm_up: warm_up})
    #     else:
    #         cost, latent_loss = self.sess.run((self.cost, self.latent_loss),
    #                                           feed_dict={self.n_z_samples: self.samples,
    #                                                      self.x: X,
    #                                                      self.x_target : X_target,
    #                                                      self._warm_up: warm_up})
    #
    #     return cost, latent_loss
    #
    # def old_create_dataset_nodes(self, dataset):
    #     # TODO move to dataset.isPerturbed()
    #     perturbed_train_set = getattr(dataset, "perturbed_train_set_x", None)
    #     perturbed_test_set = getattr(dataset, "perturbed_test_set_x", None)
    #
    #     assert((perturbed_train_set is not None and perturbed_test_set is not None)
    #            or (perturbed_train_set is None and perturbed_test_set is None),
    #            "Either both train and test perturbed are None, or both train and perturbed must be specific in the dataset")
    #
    #     # get train_set and test_set
    #     if (perturbed_train_set is not None and perturbed_test_set is not None):
    #         perturbed_dataset = 1
    #     else:
    #         perturbed_dataset = 0
    #
    #     # what I will do next, is to move from
    #     #     train_set_x, perturbed_train_set_x
    #     # which are obtained from the dataset, to
    #     #     train_set, preocessed_train_set
    #     # based on the value of perturbed_dataset, stochastic and binary
    #
    #     if perturbed_dataset:
    #         (train_set_x, perturbed_train_set_x, _) = dataset.get_training_set_batch(self.batch_size_train, returnPerturbed=perturbed_dataset)
    #         (test_set_x, perturbed_test_set_x, _) = dataset.get_test_set_batch(self.batch_size_test, returnPerturbed=perturbed_dataset)
    #         train_set = perturbed_train_set_x
    #         test_set = perturbed_test_set_x
    #     else:
    #         (train_set_x, _) = dataset.get_training_set_batch(self.batch_size_train, returnPerturbed=perturbed_dataset)
    #         (test_set_x, _) = dataset.get_test_set_batch(self.batch_size_test, returnPerturbed=perturbed_dataset)
    #         train_set = train_set_x
    #         test_set = test_set_x
    #
    #     if self.stochastic:
    #         if self.binary:
    #             augmented_train_set = utils.tf_sample_discrete_from_continuous(train_set)
    #             augmented_test_set = utils.tf_sample_discrete_from_continuous(test_set)
    #         else:
    #             # TODO here I suppose the input are in 0,1
    #             # TODO you may check whether to clip or not based on the domain of the data
    #             augmented_train_set, noise_train = utils.tf_add_gaussian_noise_and_clip(train_set,self.stochastic_noise_param, low=0, high=1)
    #             augmented_test_set, noise_test = utils.tf_add_gaussian_noise_and_clip(test_set,self.stochastic_noise_param, low=0, high=1)
    #
    #         processed_train_set = augmented_train_set
    #         processed_test_set = augmented_test_set
    #
    #     else:
    #         processed_train_set = train_set
    #         processed_test_set = test_set
    #
    #     # now we need to compute target_train_set, based on denoising
    #
    #     if self.binary:
    #         if perturbed_dataset:
    #             raise Exception("Not implemented for the moment")
    #         else:
    #             target_train_set = train_set_x
    #             target_test_set = test_set_x
    #             if not self.denoising and self.stochastic:
    #                 # use the same noise for continuous datasets
    #                 target_train_set = processed_train_set
    #                 target_test_set = processed_test_set
    #     else:
    #         target_train_set = train_set_x
    #         target_test_set = test_set_x
    #         if not self.denoising and self.stochastic:
    #             # use the same noise for continuous datasets
    #             target_train_set = utils.tf_clip(train_set_x + noise_train)
    #             target_test_set = utils.tf_clip(test_set_x + noise_test)
    #
    #     '''
    #     # TODO dropout could be obtained with a dropout layer, ignore for the moment
    #     # add drop-out noise:
    #     if drop_out>0:
    #         drop = np.random.binomial(1, 1 - drop_out, size=X.shape)
    #         X = X * drop
    #         X_target = X_target * drop
    #     '''
    #     '''
    #      TODO bit-flip for the moment is ignored
    #     # add random bit-flip:
    #     if bit_flip>0:
    #         flip = np.random.binomial(1, bit_flip, size=X.shape)
    #         X_target = (X_target + flip) % 2
    #         X = (X + flip) % 2
    #     '''
    #
    #     # TODO do I need to clip for dinary dataset?
    #
    #     # these are the nodes for the train_set
    #     x_train = processed_train_set
    #     x_train_target = target_train_set
    #     x_test = processed_test_set
    #     x_test_target = target_test_set
    #
    #     # we need to shrink data to make sure that we don't have 0 and 1 for continuous
    #     # data, otherwise we may have "division by 0" for the likelihood of the logit-normal
    #     # also, this may be usefu for binary data, to avoid gradients equal to zero in the first
    #     # layer
    #     #
    #     # if self.binary==0 and self.synthetic==0:
    #
    #     if self.binary==0:
    #         # TODO add a check that the domain is in [0,1]
    #         x_train = utils.tf_rescale(x_train, self.rescale)
    #         x_train_target = utils.tf_rescale(x_train_target, self.rescale)
    #         x_test = utils.tf_rescale(x_test, self.rescale)
    #         x_test_target = utils.tf_rescale(x_test_target, self.rescale)
    #
    #     return x_train, x_train_target, x_test, x_test_target

    # #TODO-ARGO2 (Riccardo) can't we really do without this replication for the loss... ?
    # def _replicate_x(self, x, x_target):
    #     # TODO (Luigi) move it from here, only used in VAECostFunction
    #     # TODO port the "replicate" to Distribution
    #     # TODO-ARGO2 how would this change with inputs of arbitrary shape?
    #
    #     if len(self.x_shape) > 1:
    #         raise ValueError("# TODO-ARGO2 how would this change with inputs of arbitrary shape?")
    #     input_size = self.x_shape[0]
    #
    #     if self.binary==1:
    #         # here we do cross entropy and we need three dimensions
    #         # due to the implementation in TF of sigmoid_cross_entropy_with_logits()
    #         x_replicate = tf.reshape(tf.tile(x, [1, self.n_z_samples]), [-1, self.n_z_samples, input_size])
    #         x_target_replicate = tf.reshape(tf.tile(x_target, [1, self.n_z_samples]), [-1, self.n_z_samples, input_size])
    #     else:
    #         # here we use a Gaussian distribution and "two dimensions are enough" (Alexandra)
    #         # TODO
    #         x_replicate = tf.reshape(tf.tile(x, [1, self.n_z_samples]), [-1, input_size])
    #         x_target_replicate = tf.reshape(tf.tile(x_target, [1, self.n_z_samples]), [-1, input_size])
    #     return x_replicate, x_target_replicate
    #
