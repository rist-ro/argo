import time

import matplotlib
import numpy as np
import tensorflow as tf

from .PCALatentVariablesHook import PCALatentVariablesHook
from .TwoDimPCALatentVariablesHook import TwoDimPCALatentVariablesHook
from .WavClusterAnomalyDetectionHook import WavClusterAnomalyDetectionHook
from .WavGenerateHook import WavGenerateHook
from .WavLatentPCAHook import WavLatentPCAHook
from .WavReconstructHook import WavReconstructHook
from .WavenetGaussianVisualizationHook import WavenetGaussianVisualizationHook
from .WavenetVAENetwork import WavenetVAENetwork

matplotlib.use('Agg')

from argo.core.network.AbstractAutoEncoder import AbstractAutoEncoder
from .wavenet.utils import empty_all_queues, mu_law
from .wavenet.mel_utils_tensorflow import mfcc, mcd
from argo.core.CostFunctions import CostFunctions

from argo.core.hooks.LoggingMeanTensorsHook import LoggingMeanTensorsHook

from datasets.Dataset import TRAIN_LOOP, TRAIN, VALIDATION
from argo.core.argoLogging import get_logger

tf_logging = get_logger()


class WavenetVAE(AbstractAutoEncoder):
    """WavenetVAE
    ...........

    """

    launchable_name = "WAVENET_VAE"

    default_params = {
        **AbstractAutoEncoder.default_params,

        # "cost_function": ("WavenetAECostFunction", {}),
        "epochs": 2000,  # number of epochs
    }

    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):
        # NB need to create the network before the super().__init__ because id generation depends on the network
        self._network = WavenetVAENetwork(opts)
        self._cost_function = CostFunctions.instantiate_cost_function(opts["cost_function"])
        super().__init__(opts, dirName, check_ops, gpu, seed)

        self.x_reconstruction_distr_tf = None
        self.x_reconstruction_node_tf = None
        self.x_reconstruction_distr = None
        self.x_reconstruction_node = None
        self.z = None
        self.x_shifted_qs = None
        self.n_z_samples = None
        self.reconstr_loss_t = None

    def create_id(self):
        _id = self.launchable_name

        _id += '-c' + self._cost_function.create_id(self._opts["cost_function"][1])

        super_id = super().create_id()
        network_id = self._network.create_id()

        _id += super_id + network_id

        return _id

    def create_hooks(self, config):
        hooks = super().create_hooks(config)

        # LOGGING HOOKS

        # the shape of the input could be different from train to eval
        dim_with_channels = tf.cast(tf.reduce_prod(tf.shape(self.x)[1:]),
                                    tf.float32)

        # check https://www.reddit.com/r/MachineLearning/comments/56m5o2/discussion_calculation_of_bitsdims/
        bits_dim = (self.loss / dim_with_channels) / tf.log(2.0)  # - tf.log(256.0)

        psnr_reconstruction_quality = tf.image.psnr(self.x, self.x_reconstruction_node_tf, max_val=tf.reduce_max(self.x))

        sample_rate = self.dataset.sample_rate
        mfcc_original = mfcc(self.x, samplerate=sample_rate, preemph=0)
        mfcc_reconstructed = mfcc(self.x_reconstruction_node_tf, samplerate=sample_rate, preemph=0)
        mcd_reconstruction_quality = mcd(mfcc_original, mfcc_reconstructed)

        tensors_to_average = [
            [[psnr_reconstruction_quality]
             ],
            [[mcd_reconstruction_quality]
             ],
            [[-self.loss],
             ],
            [[bits_dim],
             ],
            self.nodes_to_track
        ]

        tensors_to_average_names = [
            [['PSNR_reconstruction_quality']
             ],
            [['MCD_reconstruction_distortion']
             ],
            [["LB_log(p)"]
             ],
            [["b/d"]
             ],
            self.names_nodes_to_track
        ]

        tensors_to_average_plots = [
            [{
                 "fileName": 'psnr_reconstr_quality'}
             ],
            [{
                "fileName": 'mcd_reconstr_distortion'}
             ],
            [{
                 "fileName": "loss"}
             ],
            [{
                 "fileName": "bits_dim"}
             ],
            self.loss_nodes_to_log_filenames
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

        kwargs = config.get("WavReconstructHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(WavReconstructHook(model=self,
                                            dirName=self.dirName,
                                            **kwargs
                                            ))

        kwargs = config.get("WavenetGaussianVisualizationHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(WavenetGaussianVisualizationHook(model=self,
                                                          dirName=self.dirName,
                                                          **kwargs
                                                          ))

        kwargs = config.get("WavGenerateHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(WavGenerateHook(model=self,
                                         dirName=self.dirName,
                                         **kwargs
                                         ))

        kwargs = config.get("TwoDimPCALatentVariablesHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(TwoDimPCALatentVariablesHook(model=self,
                                                      dirName=self.dirName,
                                                      tensors=[self.z],
                                                      tensors_names=['z'],
                                                      datasets_keys=[TRAIN, VALIDATION],
                                                      # don't change the order (Luigi)
                                                      **kwargs
                                                      )
                         )

        kwargs = config.get("PCALatentVariablesHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(PCALatentVariablesHook(model=self,
                                                dirName=self.dirName,
                                                tensors=[tf.reshape(self.z, [-1, self.z.shape[-1]])],
                                                tensors_names=['z'],
                                                datasets_keys=[TRAIN, VALIDATION],  # don't change the order (Luigi)
                                                **kwargs
                                                )
                         )

        kwargs = config.get("WavLatentPCAHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(WavLatentPCAHook(model=self,
                                                dirName=self.dirName,
                                                dataset_keys=[TRAIN, VALIDATION],  # don't change the order (Luigi)
                                                **kwargs
                                                )
                         )

        kwargs = config.get('WavClusterAnomalyDetectionHook', None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(WavClusterAnomalyDetectionHook(model=self,
                                                dirName=self.dirName,
                                                dataset_keys=[TRAIN, VALIDATION],  # don't change the order (Luigi)
                                                **kwargs
                                                )
                         )


        return hooks

    def create_network(self):
        # create autoencoder network
        # self.x_shifted, self.z, self.x_reconstruction_distr, self.x_reconstruction_node, self.x_gen = self._network(self.x)
        net = self._network(self.x, x_shape=self.dataset.x_shape_train)
        self.x_shifted_qs = net["x_shifted_qs"]
        self.z = net["z"]
        self._upsample_encoding = net['upsample_encoding']
        self.z_upsampled = net['z_upsampled']
        self.samples_posterior = net['samples_posterior']

        self.x_reconstruction_distr_tf = net["x_rec_distr_tf"]
        self.x_reconstruction_logits_tf = self.x_reconstruction_distr_tf.logits
        self.x_reconstruction_node_tf = net["x_rec_tf"]

        self.x_t = net["x_t"]
        self.x_t_qs = net["x_t_qs"]
        self.z_t = net["z_t"]
        self.x_tp1_distr = net["x_tp1_distr"]
        self.x_tp1 = net["x_tp1"]
        self.x_target_t = tf.placeholder(tf.float32, shape=[None, 1], name='x_target_t')
        self.queues_init_ops = net["queues_init_ops"]
        self.queues_push_ops = net["queues_push_ops"]
        self.queues_dicts = net["queues_dicts"]

        self.n_z_samples = net["n_z_samples"]

        self.queues_dequeue = [qd["dequeue"] for qd in self.queues_dicts]
        self.queues_size = [qd["size"] for qd in self.queues_dicts]
        self._gaussian_model_latent = net["latent"]
        self._gaussian_model_latent_mean = self._gaussian_model_latent.mean()
        self._gaussian_model_latent_mean_transp = tf.transpose(self._gaussian_model_latent_mean, perm=[0, 2, 1])
        self._gaussian_model_latent_covariance = self._gaussian_model_latent.covariance()
        self._prior = net["prior"]
        self._prior_mean = self._prior.mean()
        self._prior_covariance = self._prior.covariance()

        self._model_visible = self.x_reconstruction_distr_tf
        self.x_target_quantized = mu_law(self.x_target)
        self.x_target_quantized_squeezed = tf.placeholder(tf.float32, shape=[None, None])

    def create_loss(self):
        # A TFDeepLearningModel model should be free to specify a model dependent loss..
        # compute the cost function, passing the model as a parameter
        self.loss, self.nodes_to_track, self.names_nodes_to_track, self.loss_nodes_to_log_filenames = self._cost_function(
            self)

        self.reconstr_loss_t = self._cost_function.reconstr_loss_per_sample(self.x_target_t, self.n_z_samples, self.x_tp1_distr)
        self.reconstr_loss_tf_per_sample = self._cost_function.reconstr_loss_per_sample(self.x_target_quantized_squeezed, self.n_z_samples, self._model_visible)
        self.reconstr_loss_tf_val = self._cost_function.reconstruction_loss(self.x_target_quantized_squeezed, self.n_z_samples, self._model_visible)

    def encode(self, X, sess=None):
        """Encode data by mapping it into the latent space."""

        sess = sess or self.get_raw_session()
        bs, length, ch = X.shape
        bs_train = self.batch_size['train']

        if self._network._e_hidden_channels > 128 and bs > bs_train:
            zs, means, covs, prior_means, prior_covs, x_shifted_list = [], [], [], [], [], []
            start_batch, end_batch = 0, bs_train

            while start_batch != end_batch:
                z, mean, cov, prior_mean, prior_cov, x_shifted = self.encode(X[start_batch:end_batch])
                zs.append(z)
                means.append(mean)
                covs.append(cov)
                prior_means.append(prior_mean)
                prior_covs.append(prior_cov)
                x_shifted_list.append(x_shifted)

                start_batch = end_batch
                end_batch = min(bs, end_batch + bs_train)

            return np.concatenate(zs, axis=0), np.concatenate(means, axis=0), np.concatenate(covs, axis=0),\
                   np.concatenate(prior_means, axis=0), np.concatenate(prior_covs, axis=0), np.concatenate(x_shifted_list, axis=0)
        else:
            return sess.run([self.z, self._gaussian_model_latent_mean_transp, self._gaussian_model_latent_covariance,
                             self._prior_mean, self._prior_covariance, self.x_shifted_qs],
                            feed_dict={
                                self.x: X})

    def get_mean_covariance(self, X, sess=None):
        '''
        returns the mean and the covariance of the latent gaussian distribution
        '''
        sess = sess or self.get_raw_session()
        return sess.run([self._gaussian_model_latent_mean_transp, self._gaussian_model_latent_covariance],
                        feed_dict={
                            self.x: X
                        })

    # def decode(self, Z, sess=None):
    #     """Decode latent vectors in input data."""
    #
    #     sess = sess or self.get_raw_session()
    #     import pdb;pdb.set_trace()
    #     return sess.run(self.x_gen,
    #                     feed_dict={ self.z : Z })
    #                         # self.x_shifted: X_shifted})

    def decode_tf(self, Z, X_shifted, x_target_quantized, sess=None, batch_size=None):
        """Decode latent vectors in input data for an autoregressive model,
        with teacher forcing.
            Args:
                x_target_quantized (np.array): the target signal with values in range [0, 256]
                                                and shape (bs, signal_length)
            Returns:
                reconstruction from Z and the associated reconstruction loss
        """

        sess = sess or self.get_raw_session()

        # decode in mini_batches because if we decode more than 50 samples at a time with 6000x8 length then
        # we will run out of memory
        total_batch, signal_length = x_target_quantized.shape
        mini_batch_size = min(batch_size or total_batch, total_batch)
        start_batch, end_batch = 0, mini_batch_size

        reconstruction = np.zeros((total_batch, signal_length, 1), dtype='float32')
        reconstr_losses = np.zeros((total_batch), dtype='float32')

        # have to do this in batches because for total_batch > 50 it will run out of memory
        while start_batch != end_batch:
            reconstruction[start_batch:end_batch], reconstr_losses[start_batch:end_batch] = \
                sess.run([self.x_reconstruction_node_tf, self.reconstr_loss_tf_per_sample],
                         feed_dict={
                             self.z: Z[start_batch:end_batch],
                             self.x_shifted_qs: X_shifted[start_batch:end_batch],
                             self.x_target_quantized_squeezed: x_target_quantized[start_batch:end_batch]
                         })

            start_batch = end_batch
            end_batch = min(end_batch + mini_batch_size, total_batch)

        return reconstruction, reconstr_losses

    def reconstruct_tf(self, X, sess=None):
        """ Use AE to reconstruct given data.
        With teacher forcing.
        """
        sess = sess or self.get_raw_session()

        return sess.run(self.x_reconstruction_node_tf,
                        feed_dict={
                            self.x: X})

    def decode(self, Z, X_0=None, input_qs=False, sess=None, x_target_quantized=None):
        """Decode latent vectors in input data for wavenet autoregressive model
            Args:
                x_target_quantized (np.array): the target signal with values in range [0, 256]
                                                and shape (bs, signal_length)

            Returns:
                the reconstruction without teacher forcing from Z and the associated reconstruction loss computed
                at each time step individually
        """
        sess = sess or self.get_raw_session()

        bs, z_length, z_ch = Z.shape
        hop_length = self._network._hop_length
        time_length = z_length * hop_length

        if self._upsample_encoding:
            Z = sess.run(self.z_upsampled, feed_dict={self.z: Z})
            hop_length = 1

        # assume x single channel
        if X_0 is None:
            X_0 = np.zeros([bs, 1, 1])

        # beware here we can input x_t or x_t_qs, maybe x_t_qs ia better but we should test this
        # inputs_node = self.x_t
        if input_qs:
            inputs_node = self.x_t_qs
        else:
            inputs_node = self.x_t

        # initialize the fast generation queues with all 0s, x_t is needed for the batch_size
        sess.run(self.queues_init_ops, feed_dict={
            inputs_node: X_0})

        audio_batch = np.zeros((bs, time_length, 1), dtype=np.float32)
        x = X_0

        reconstr_loss_per_sample = np.zeros((bs, time_length), dtype=np.float32)

        before = time.time()
        for t in range(time_length):
            i_z = t // hop_length

            x, reconstr_loss_t = sess.run([self.x_tp1, self.reconstr_loss_t, self.queues_push_ops],
                                          feed_dict={
                                         inputs_node: x,
                                         self.z_t: Z[:, i_z:i_z + 1, :],
                                         self.x_target_t: x_target_quantized[:, t:t + 1]
                                     }
                                          )[0:2]
            audio_batch[:, t, 0] = x[:, 0, 0]
            reconstr_loss_per_sample[:, t] = reconstr_loss_t

            if t % 1000 == 0:
                tf_logging.info("Sample: %d" % t)
        after = time.time()
        diff = after - before
        tf_logging.info("decoding time is %s:%s" % (str(int(diff // 60)), str(diff % 60).zfill(2)))

        # NB(Riccardo) I dequeue all the elements from the queues, otherwise next call to queues_init_ops will hang indefinitely
        # for faster dequeuing the batch size should be fixed, to be able to use q.dequeue_many(qsize)
        before = time.time()
        all_sizes = empty_all_queues(sess, self.queues_size, self.queues_dequeue)
        after = time.time()
        diff = after - before
        tf_logging.info("dequeuing time is %s:%s" % (str(int(diff // 60)), str(diff % 60).zfill(2)))

        # naive dequeuing (slow)
        # all_sizes = []
        # for qs, qd in zip(:
        #     qsize = sess.run(dq["size"])
        #     all_sizes.append(dq["size"])
        #     for i in range(qsize):
        #         sess.run(dq["dequeue"])

        assert all(all_sizes == 0), "queues not succefully emptied after decoding"

        return audio_batch, np.sum(reconstr_loss_per_sample, axis=1)

    def mean_reconstr_loss(self, log_probs):
        '''This method is used in WavGenerateHook in order to be able to compute the loss separately for hs, zs train
            and validation. This is because we decode all of the above samples at once to reduce time
        '''
        # now (ready for arbitrary intermediate samplings)
        all_axis_but_first = list(range(len(log_probs.shape)))[1:]  # [1]
        # independent p for each input pixel
        log_p = np.apply_over_axes(np.sum, log_probs, axes=all_axis_but_first)  # 4
        log_p = np.squeeze(log_p)
        # average over all the samples and the batch (they are both stacked on the axis 0)
        mean_reconstr_loss = np.mean(log_p, axis=0)

        return mean_reconstr_loss

    def decode_debug_astf(self, Z, X_shifted, sess=None):
        """Decode latent vectors in input data for wavenet autoregressive model,
        test of the fast generation to see if it is equal to the tf. It should be."""
        sess = sess or self.get_raw_session()

        bs, z_length, z_ch = Z.shape
        hop_length = self._network._hop_length
        time_length = z_length * hop_length

        assert X_shifted.shape == (bs, time_length, 1)

        # initialize the fast generation queues with all 0s, x_t is needed for the batch_size
        sess.run(self.queues_init_ops, feed_dict={
            self.x_t_qs: X_shifted[:, :1, :]})

        audio_batch = np.zeros((bs, time_length, 1), dtype=np.float32)

        log_probs = np.zeros((bs, time_length))

        before = time.time()
        for t in range(time_length):
            i_z = t // hop_length
            x, log_prob_t = sess.run([self.x_tp1, self.reconstr_loss_t, self.queues_push_ops],
                                     feed_dict={
                                         self.x_t_qs: X_shifted[:, t:t + 1, :],
                                         self.z_t: Z[:, i_z:i_z + 1, :]
                                         # self.x_target_t: x_target_
                                     }
                                     )[0:2]
            audio_batch[:, t, 0] = x[:, 0, 0]
            log_probs[:, t:t + 1] = log_prob_t
            if t % 1000 == 0:
                tf_logging.info("Sample: %d" % t)
        after = time.time()
        diff = after - before
        tf_logging.info("decoding time is %s:%s" % (str(int(diff // 60)), str(diff % 60).zfill(2)))

        # NB(Riccardo) I dequeue all the elements from the queues, otherwise next call to queues_init_ops will hang indefinitely
        # for faster dequeuing the batch size should be fixed, to be able to use q.dequeue_many(qsize)

        # smart dequeuing contemporaneously on all queues till I can
        before = time.time()
        all_sizes = empty_all_queues(sess, self.queues_size, self.queues_dequeue)
        after = time.time()
        diff = after - before
        tf_logging.info("dequeuing time is %s:%s" % (str(int(diff // 60)), str(diff % 60).zfill(2)))

        assert all(all_sizes == 0), "queues not succefully emptied after decoding"

        return audio_batch, self.mean_reconstr_loss(log_probs)

    def reconstruct(self, X, sess=None):
        """ Use AE to reconstruct given data.
        With teacher forcing.
        """
        sess = sess or self.get_raw_session()

        Z, _ = self.encode(X, sess=sess)

        return self.decode(Z, sess=sess)

    def generate(self, batch_size=1, sess = None):
        pass

