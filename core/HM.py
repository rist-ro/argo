import matplotlib
import tensorflow as tf

from argo.core.hooks.ImportanceSamplingHook import ImportanceSamplingHook
from argo.core.hooks.ImagesInputHook import ImagesInputHook
from argo.core.optimizers import NrSamples
from core.hooks.HMFisherMatrixHook2 import HMFisherMatrixHook2
from core.hooks.ImageReconstructHook import ImagesReconstructHook
from core.hooks.ThreeByThreeHook import ThreeByThreeHook

PHASE_WAKE = "wake"
PHASE_SLEEP = "sleep"

matplotlib.use('Agg')

from argo.core.network.AbstractAutoEncoder import AbstractAutoEncoder

from argo.core.CostFunctions import CostFunctions

from core.networks.HMNetwork import HMNetwork

from argo.core.hooks.LoggingMeanTensorsHook import LoggingMeanTensorsHook
from core.hooks.ImagesGenerateHook import ImagesGenerateHook

from datasets.Dataset import TRAIN_LOOP, VALIDATION, TRAIN, TEST

NUMTOL = 1e-7


class HM(AbstractAutoEncoder):
    """ Helmholtz Machine
    """

    launchable_name = "HM"

    default_params = {
        **AbstractAutoEncoder.default_params,

        "samples": 10,
        "pm_one":  True,
        "epochs":  2000,  # number of epochs
    }

    def create_id(self):

        _id = self.launchable_name

        # add to the ID the information of the cost function
        _id += '-c' + self._cost_function.create_id(self._opts["cost_function"])

        _id += '-s' + NrSamples.get_n_s_id(self._opts["samples"])
        _id += '-cp' + str(self._opts["clip_probs"])
        if not self._opts["pm_one"]:
            _id += '-pm' + str(self._opts["pm_one"])

        super_id = super().create_id()
        network_id = self._network.create_id()

        _id += super_id + network_id
        return _id

    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):
        # NB need to create the network before the super init because id generation depends on the network
        self._pm_one = (opts["pm_one"] if "pm_one" in opts else self.default_params["pm_one"])
        self._clip_probs = opts["clip_probs"]
        self._network = HMNetwork(opts, self._clip_probs, pm=self._pm_one, name="hhm_network")

        self._cost_function = CostFunctions.instantiate_cost_function(opts["cost_function"],
                                                                      module_path="helmholtz-machine")
        super().__init__(opts, dirName, check_ops, gpu, seed)

        self.samples = self._opts["samples"]

        # important nodes
        self._gaussian_model_latent = None
        self._model_visible = None
        self.x_reconstruction_node = None

    def create_hooks(self, config):
        hooks = super().create_hooks(config)

        tensors_to_average = [
            [[self.cost]], [[self._prior.mean()],[self.n_z_samples]],

        ]
        tensors_to_average_names = [
            [["loss_wake"]], [["prior_mean"],["samples"]
                              ],

        ]
        tensors_to_average_plots = [
            [
                {
                    "fileName": "loss"},
            ],
            [
                {
                    "fileName": "mean"},
                {
                    "fileName": "samples"}
            ]

        ]

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
                                            datasets_keys=[VALIDATION],
                                            time_reference=self._time_reference_str
                                            )
                     )

        tensors_to_average = [
            self.loss_nodes_to_log,
        ]
        tensors_to_average_names = [
            self.loss_nodes_to_log_names,
        ]
        tensors_to_average_plots = [
            self.loss_nodes_to_log_filenames,
        ]

        hooks.append(LoggingMeanTensorsHook(model=self,
                                            fileName="log2",
                                            dirName=self.dirName,
                                            tensors_to_average=tensors_to_average,
                                            tensors_to_average_names=tensors_to_average_names,
                                            tensors_to_average_plots=tensors_to_average_plots,
                                            average_steps=self._n_steps_stats,
                                            tensorboard_dir=self._tensorboard_dir,
                                            print_to_screen=False,
                                            trigger_summaries=config["save_summaries"],
                                            plot_offset=self._plot_offset,
                                            train_loop_key=TRAIN_LOOP,
                                            datasets_keys=[],
                                            time_reference=self._time_reference_str
                                            )
                     )

        kwargs = config.get("HMFisherMatrixHook2", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}

            hooks.append(HMFisherMatrixHook2(model=self,
                                             dirName=self.dirName,
                                             **kwargs)
                         )
        else:
            if self._optimizer._d_p is not None:
                t_to_average = [
                    [[self._optimizer._diagonal_pad]]
                ]
                t_to_average_names = [
                    [["diagonal_pad"]],
                ]
                t_to_average_plots = [
                    [{
                        "fileName": "diagonal_pad"}]
                ]
                hooks.append(LoggingMeanTensorsHook(model=self,
                                                    fileName="dp",
                                                    dirName=self.dirName,
                                                    tensors_to_average=t_to_average,
                                                    tensors_to_average_names=t_to_average_names,
                                                    tensors_to_average_plots=t_to_average_plots,
                                                    average_steps=self._n_steps_stats,
                                                    tensorboard_dir=self._tensorboard_dir,
                                                    trigger_summaries=config["save_summaries"],
                                                    print_to_screen=False,
                                                    plot_offset=self._plot_offset,
                                                    train_loop_key=TRAIN_LOOP,
                                                    datasets_keys=[]
                                                    )
                             )

        kwargs = config.get("ImagesGenerateHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}

            hooks.append(ImagesGenerateHook(model=self,
                                            dirName=self.dirName,
                                            pm_one=self._pm_one,
                                            **kwargs
                                            )
                         )
        kwargs = config.get("ImagesInputHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}

            hooks.append(ImagesInputHook(model=self,
                                         dirName=self.dirName,
                                         pm_one=self._pm_one,
                                         **kwargs
                                         )
                         )

        kwargs = config.get("ImagesReconstructHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}

            hooks.append(ImagesReconstructHook(model=self,
                                               dirName=self.dirName,
                                               pm_one=self._pm_one,
                                               **kwargs)
                         )

        kwargs = config.get("ThreeByThreeHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}

            hooks.append(ThreeByThreeHook(model=self,
                                          tensorboard_dir=self._tensorboard_dir,
                                          dirName=self.dirName,
                                          **kwargs)
                             )

        kwargs = config.get("LogpImportanceSamplingHook", None)
        if kwargs:
            if not isinstance(kwargs, list):
                kwargs = [kwargs]
            for kw in kwargs:
                kws = {**self._plot_model_hooks_kwargs,
                       **kw}
                hooks.append(ImportanceSamplingHook(model = self,
                                                    dirName = self.dirName,
                                                    tensors_to_average = [self.importance_sampling_node],
                                                    datasets_keys = [TRAIN, VALIDATION, TEST],
                                                    **kws
                                                   )
                            )
        return hooks

    def _create_gradient_hook(self, config):

        # gradienthook
        tensors_to_average = [
            [[self.gradient_weight_global_norms[PHASE_WAKE][0]],
             self.gradient_norms[PHASE_WAKE],
             ],
            [
                [self.gradient_weight_global_norms[PHASE_SLEEP][0]],
                self.gradient_norms[PHASE_SLEEP]
            ],
            [[self.gradient_weight_global_norms[PHASE_WAKE][1]],
             self.weight_norms[PHASE_WAKE],
             ],
            [
                [self.gradient_weight_global_norms[PHASE_SLEEP][1]],
                self.weight_norms[PHASE_SLEEP]
            ],
        ]

        layer_names_wake = ["L" + name.split(":")[0] for name in self.gradient_names[PHASE_WAKE]]

        layer_names_sleep = ["L" + name.split(":")[0] for name in self.gradient_names[PHASE_SLEEP]]

        tensors_to_average_names = [
            [["gradient_global_norms_wake"],
             layer_names_wake,
             ],
            [
                ["gradient_global_norms_sleep"],
                layer_names_sleep
            ],
            [["weight_global_norms_wake"],
             layer_names_wake,
             ],
            [
                ["weight_global_norms_sleep"],
                layer_names_sleep
            ],
        ]

        tensors_to_average_plots = [
            [{
                "fileName":      "gradient_global_norms_wake",
                "logscale-y":    1,
                "compose-label": 0},
                {
                    "fileName":      "gradient_norms_wake",
                    "logscale-y":    1,
                    "compose-label": 0}, ],
            [
                {
                    "fileName":      "gradient_global_norms_sleep",
                    "logscale-y":    1,
                    "compose-label": 0},
                {
                    "fileName":      "gradient_norms_sleep",
                    "logscale-y":    1,
                    "compose-label": 0}
            ],
            [{
                "fileName":      "weight_global_norms_wake",
                "logscale-y":    1,
                "compose-label": 0},
                {
                    "fileName":      "weight_norms_wake",
                    "logscale-y":    1,
                    "compose-label": 0}, ],
            [
                {
                    "fileName":      "weight_global_norms_sleep",
                    "logscale-y":    1,
                    "compose-label": 0},
                {
                    "fileName":      "weight_norms_sleep",
                    "logscale-y":    1,
                    "compose-label": 0}
            ],
        ]

        kwargs = config.get("GradientsHook", None)
        hook = None
        if kwargs:
            gradient_period = config["GradientsHook"]["period"]
            gradient_steps = self._get_steps(gradient_period, self._time_reference_str)
            hook = LoggingMeanTensorsHook(model=self,
                                          fileName="gradient",
                                          dirName=self.dirName,
                                          tensors_to_average=tensors_to_average,
                                          tensors_to_average_names=tensors_to_average_names,
                                          tensors_to_average_plots=tensors_to_average_plots,
                                          average_steps=gradient_steps,
                                          time_reference=self._time_reference_str,
                                          tensorboard_dir=self._tensorboard_dir,
                                          trigger_summaries=config["save_summaries"],
                                          # trigger_plot = True,
                                          print_to_screen=False,
                                          plot_offset=self._plot_offset,
                                          train_loop_key=TRAIN_LOOP,
                                          datasets_keys=[]
                                          )

        return hook

    def create_network(self):

        # create autoencoder network
        dictionary = self._network(self.x, self.b_size, self.n_z_samples)
        self.h_inferred, self.x_reconstruct, self.x_reconstruct_distr, self._hrw, self._hgw = dictionary[PHASE_WAKE]

        self.h_reconstruct, self.x_inferred, self.x_inferred_distr, self._hrs, self._hgs = dictionary[PHASE_SLEEP]

        x_inferred_shape = tf.shape(self.x_inferred)

        self._model_latent_mean = self._hrw[-1][0].mean()

        self._model_reconstuction_distr = self.x_reconstruct_distr
        self._model_inferred_distr = self.x_inferred_distr

        self.x_inferred_node = tf.reshape(self._model_inferred_distr.reconstruction_node(),
                                          [-1] + self.x.shape[1:].as_list())
        self.x_inferred_node = self.x_inferred_node

        self.x_inferred_sample = tf.reshape(self._model_inferred_distr.sample(),
                                            [-1] + self.x.shape[1:].as_list())
        self.x_inferred_sample = self.x_inferred_sample

        self.x_reconstruction_node = tf.reshape(self._model_reconstuction_distr.reconstruction_node(),
                                                [-1] + self.x.shape[1:].as_list())
        self.x_reconstruction_node = self.x_reconstruction_node

        self.x_reconstruction_sample = tf.reshape(self._model_reconstuction_distr.sample(),
                                                  [-1] + self.x.shape[1:].as_list())
        self.x_reconstruction_sample = self.x_reconstruction_sample

        self._prior, self._prior_samples = dictionary["prior"]

        # for the moment we always create the IS node
        self.importance_sampling_node = self._create_importance_sampling_node()

    def encode(self, X, sess=None):
        sess = sess or self.get_raw_session()

        return sess.run([self.h_inferred, self._model_latent_mean], feed_dict={
            self.n_z_samples: 1,
            self.x:           X})

    # could be merged with AE
    def decode(self, Z, sess=None):
        """Decode latent vectors in input data."""
        sess = sess or self.get_raw_session()

        return sess.run([self.x_inferred_node, self.x_inferred],
                        feed_dict={
                            self._prior_samples: Z})

    def generate(self, batch_size=1, sess=None):
        """ Generate data by sampling from latent space.

        """
        sess = sess or self.get_raw_session()

        return sess.run([self.x_inferred_node, self.x_inferred],
                        feed_dict={
                            self.b_size:      batch_size,
                            self.n_z_samples: 1})

    def reconstruct(self, X, sess=None):
        """ Use VAE to reconstruct given data. """
        return sess.run([self.x_reconstruction_node, self.x_reconstruct],
                        feed_dict={
                            self.n_z_samples: 1,
                            self.raw_x:       X})

    def get_hm_optimizer_step(self, phase, global_step=None):
        # You can use this if you want to use gradient descent
        # 1st part of minimize: compute_gradient
        if phase == PHASE_WAKE:
            # This is just for debug reasons, you may remove it later
            self.grads_and_vars_w = self._optimizer.compute_gradients(phase, self.loss, global_step=self.global_step)
            self.grads_and_vars = self.grads_and_vars_w
        else:
            self.grads_and_vars_s = self._optimizer.compute_gradients(phase, self.loss, global_step=self.global_step)
            self.grads_and_vars = self.grads_and_vars_s

        # clip gradients
        clipped_grads_and_vars = self._clip_gradients(self.grads_and_vars, self._grad_clipping_tuple)

        # compute norms in case they need to be logged
        self.gradient_names[phase] = [g.name for (g, _) in self.grads_and_vars]
        self.gradient_norms[phase] = [tf.norm(g) + NUMTOL for (g, _) in clipped_grads_and_vars]
        self.weight_norms[phase] = [tf.norm(v) + NUMTOL for (g, v) in clipped_grads_and_vars]

        # check that gradients are finite
        grads = [tf.check_numerics(g, "grads is not finite") for (g, v) in clipped_grads_and_vars]
        variables = [tf.check_numerics(v, "grads is not finite") for (g, v) in clipped_grads_and_vars]
        self.gradient_weight_global_norms[phase] = [tf.global_norm(grads), tf.global_norm(variables)]

        # 2nd part of minimize: apply_gradient, THIS IS NOT self.global_step on purpose
        optimizer_step = self._optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)

        return optimizer_step

    # reimplemented since in GAN we have two optimizers
    def set_training_op(self):

        self.gradients = {}
        self.gradient_names = {}
        self.gradient_norms = {}
        self.weight_norms = {}
        self.gradient_weight_global_norms = {}

        optimizer_step_wake = self.get_hm_optimizer_step(PHASE_WAKE)
        optimizer_step_sleep = self.get_hm_optimizer_step(PHASE_SLEEP, global_step=self.global_step)

        # grouping the training operations

        # grads = [g for g,v in [*self.grads_and_vars_w, *self.grads_and_vars_s]]
        # vars = [v for g,v in [*self.grads_and_vars_w, *self.grads_and_vars_s]]

        # import ipdb; ipdb.set_trace()
        # printy = tf.print(self.global_step,self.b_size, [(g.name, g) for g in grads], "vars\n\n", [(v.name, v) for v in vars],"norms:",tf.reduce_mean(self.gradient_norms[PHASE_SLEEP]+self.gradient_norms[PHASE_WAKE]) ,"\n\n")
        update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        self.training_op = tf.group(update_ops, optimizer_step_wake, optimizer_step_sleep)

    def create_loss(self):
        # Cross entropy loss

        cost, self.loss_nodes_to_log, self.loss_nodes_to_log_names, self.loss_nodes_to_log_filenames = self._cost_function(
            self)

        self.loss = {}
        self.loss[PHASE_WAKE] = self.loss_nodes_to_log[0][0]
        self.loss[PHASE_SLEEP] = self.loss_nodes_to_log[1][0]
        self.cost = (cost if cost is not None else tf.identity(self.loss[PHASE_WAKE]))

    def create_input_nodes(self, dataset):
        """
        creates input nodes for an autoencoder from the dataset

        Sets:
            x, x_target
        """

        super().create_input_nodes(dataset=dataset)

        self.b_size = tf.shape(self.x)[0]
        self.n_z_samples = NrSamples.process_n_samples(self.samples, self.global_epoch)

    def _create_importance_sampling_node(self):
        batch_size = self.b_size
        n_samples = self.n_z_samples
        mega_batch_size = batch_size * n_samples

        gen_layers = self._hgw
        rec_layers = self._hrw

        ps = tf.zeros((mega_batch_size))
        qs = tf.zeros((mega_batch_size))

        for i in range(len(gen_layers)):
            samples_q = rec_layers[i][1]

            distr_p = gen_layers[i][0]
            log_probs_p_all = tf.reduce_sum(distr_p.log_prob(samples_q), axis=-1)
            ps += log_probs_p_all

            if i != 0:
                distr_q = rec_layers[i][0]
                log_probs_q_all = tf.reduce_sum(distr_q.log_prob(samples_q), axis=-1)
                qs += log_probs_q_all

        # Reshape
        log_p_all = tf.reshape(ps, [n_samples, batch_size])
        log_q_all = tf.reshape(qs, [n_samples, batch_size])
        # Approximate log(p(x))
        logp_is = tf.reduce_logsumexp(log_p_all - log_q_all, axis=0)

        return logp_is
