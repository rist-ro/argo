import time

import matplotlib
import numpy as np
import tensorflow as tf

from .WavGenerateHook import WavGenerateHook
from .PCALatentVariablesHook import PCALatentVariablesHook
from .TwoDimPCALatentVariablesHook import TwoDimPCALatentVariablesHook
from .WavReconstructHook import WavReconstructHook
from .wavenet.mel_utils_tensorflow import mfcc, mcd

matplotlib.use('Agg')

from argo.core.network.AbstractAutoEncoder import AbstractAutoEncoder
from .wavenet.utils import empty_all_queues
from .WavenetAENetwork import WavenetAENetwork
from argo.core.CostFunctions import CostFunctions

from argo.core.hooks.LoggingMeanTensorsHook import LoggingMeanTensorsHook

from datasets.Dataset import TRAIN_LOOP, TRAIN, VALIDATION
from argo.core.argoLogging import get_logger
tf_logging = get_logger()


class WavenetAE(AbstractAutoEncoder):
    """WavenetAE
    ...........

    """

    launchable_name = "WAVENET_AE"

    default_params = {
        **AbstractAutoEncoder.default_params,

        "cost_function": ("WavenetAECostFunction", {}),
        "epochs":        2000,  # number of epochs
    }

    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):
        # NB need to create the network before the super().__init__ because id generation depends on the network
        self._network = WavenetAENetwork(opts)
        self._cost_function = CostFunctions.instantiate_cost_function(opts["cost_function"])
        super().__init__(opts, dirName, check_ops, gpu, seed)

        self.x_reconstruction_distr_tf = None
        self.x_reconstruction_node_tf = None
        self.x_reconstruction_distr = None
        self.x_reconstruction_node = None
        self.z = None
        self.x_shifted_qs = None
        self.loss_t = None
        self.loss_tf = None

    def create_id(self):
        _id = self.launchable_name

        _id += '-c' + self._cost_function.create_id()

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

        psnr_reconstruction_quality = tf.image.psnr(self.x, self.x_reconstruction_node_tf,
                                                    max_val=tf.reduce_max(self.x))

        sample_rate = self.dataset.sample_rate
        mfcc_original = mfcc(self.x, samplerate=sample_rate, preemph=0)
        mfcc_reconstructed = mfcc(self.x_reconstruction_node_tf, samplerate=sample_rate, preemph=0)
        mcd_reconstruction_quality = mcd(mfcc_original, mfcc_reconstructed)

        tensors_to_average = [
            [[psnr_reconstruction_quality]
             ],
            [[mcd_reconstruction_quality]
             ],
            [[self.loss],
             ],
            [*[[name] for name in self.nodes_to_track]
             ],
            [[bits_dim],
             ]
        ]

        tensors_to_average_names = [
            [['PSNR_reconstruction_quality']
             ],
            [['MCD_reconstruction_distortion']
             ],
            [["nll_loss"]
             ],
            [*[[name] for name in self.names_nodes_to_track]
             ],
            [["b/d"]
             ]
        ]

        tensors_to_average_plots = [
            [{
                "fileName": 'psnr_reconstr_quality'}
             ],
            [{
                "fileName": 'mcd_reconstr_distortion'}
             ],
            [{"fileName": "loss"}
             ],
            [{"fileName": "nodes_loss"}
             ],
            [{"fileName": "bits_dim"}
             ],
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

        kwargs = config.get("WavReconstructHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(WavReconstructHook(model=self,
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
                                                tensors=[self.z],
                                                tensors_names=['z'],
                                                datasets_keys=[TRAIN, VALIDATION],  # don't change the order (Luigi)
                                                **kwargs
                                                )
                         )

        return hooks

    def create_network(self):
        # create autoencoder network
        # self.x_shifted, self.z, self.x_reconstruction_distr, self.x_reconstruction_node, self.x_gen = self._network(self.x)
        net = self._network(self.x)

        self.x_shifted_qs = net["x_shifted_qs"]
        self.z = net["z"]
        self._upsample_encoding = net['upsample_encoding']
        self.z_upsampled = net['z_upsampled']

        self.x_reconstruction_distr_tf = net["x_rec_distr_tf"]
        self.x_reconstruction_logits_tf = self.x_reconstruction_distr_tf.logits
        self.x_reconstruction_node_tf = net["x_rec_tf"]

        self.x_t = net["x_t"]
        self.x_t_qs = net["x_t_qs"]
        self.z_t = net["z_t"]
        self.x_target_t = tf.placeholder(tf.float32, shape=(None, 1), name='x_target_t')
        self.x_target_quantized = tf.placeholder(tf.int32, shape=(None, None), name='x_target_quantized')
        self.x_tp1_distr = net["x_tp1_distr"]
        self.x_tp1 = net["x_tp1"]
        self.queues_init_ops = net["queues_init_ops"]
        self.queues_push_ops = net["queues_push_ops"]
        self.queues_dicts = net["queues_dicts"]

        self.queues_dequeue = [qd["dequeue"] for qd in self.queues_dicts]
        self.queues_size = [qd["size"] for qd in self.queues_dicts]

    def create_loss(self):
        # A TFDeepLearningModel model should be free to specify a model dependent loss..
        # compute the cost function, passing the model as a parameter
        self.loss, self.nodes_to_track, self.names_nodes_to_track = self._cost_function(self)

        self.loss_t = self._cost_function.loss_per_sample(self.x_tp1_distr.logits, self.x_target_t)
        self.loss_tf = self._cost_function.loss_per_sample(self.x_reconstruction_logits_tf, self.x_target_quantized)

    def encode(self, X, sess=None):
        """Encode data by mapping it into the latent space."""

        # sess = sess or self.get_raw_session()
        #
        # return sess.run([self.z, self.x_shifted_qs], feed_dict={
        #     self.x: X})
        sess = sess or self.get_raw_session()
        bs, length, ch = X.shape
        bs_train = self.batch_size['train']

        if self._network._e_hidden_channels > 128 and bs > bs_train:
            zs, x_shifted_list = [], []
            start_batch, end_batch = 0, bs_train

            while start_batch != end_batch:
                z, x_shifted = self.encode(X[start_batch:end_batch])
                zs.append(z)
                x_shifted_list.append(x_shifted)

                start_batch = end_batch
                end_batch = min(bs, end_batch + bs_train)

            return np.concatenate(zs, axis=0), np.concatenate(x_shifted_list, axis=0)
        else:
            return sess.run([self.z, self.x_shifted_qs],
                            feed_dict={
                                self.x: X})

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

        total_batch, signal_length = x_target_quantized.shape
        mini_batch_size = min(batch_size or total_batch, total_batch)
        start_batch, end_batch = 0, mini_batch_size

        reconstruction = np.zeros((total_batch, signal_length, 1))
        reconstr_losses = np.zeros((total_batch))

        while start_batch != end_batch:
            reconstruction[start_batch:end_batch], reconstr_loss = sess.run([self.x_reconstruction_node_tf, self.loss_tf],
                            feed_dict={self.z : Z[start_batch:end_batch],
                                       self.x_shifted_qs: X_shifted[start_batch:end_batch],
                                       self.x_target_quantized: x_target_quantized[start_batch:end_batch]
                                       })
            reconstr_losses[start_batch:end_batch] = np.mean(reconstr_loss, axis=1)
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

        #beware here we can input x_t or x_t_qs, maybe x_t_qs ia better but we should test this
        # inputs_node = self.x_t
        if input_qs:
            inputs_node = self.x_t_qs
        else:
            inputs_node = self.x_t

        # initialize the fast generation queues with all 0s, x_t is needed for the batch_size
        sess.run(self.queues_init_ops, feed_dict={inputs_node : X_0})

        audio_batch = np.zeros((bs, time_length, 1), dtype=np.float32)
        x = X_0

        losses = []

        before = time.time()
        for t in range(time_length):
            i_z = t // hop_length
            x, loss_t = sess.run([self.x_tp1, self.loss_t, self.queues_push_ops],
                                 feed_dict={inputs_node: x,
                                            self.z_t: Z[:, i_z:i_z + 1, :],
                                            self.x_target_t: x_target_quantized[:, t:t + 1]
                                            }
                                 )[0:2]
            audio_batch[:, t, 0] = x[:, 0, 0]
            losses.append(np.mean(loss_t))

            if t % 1000 == 0:
                tf_logging.info("Sample: %d" % t)
        after = time.time()
        diff = after - before
        tf_logging.info("decoding time is %s:%s" % (str(int(diff // 60)), str(diff % 60).zfill(2)))

        #NB(Riccardo) I dequeue all the elements from the queues, otherwise next call to queues_init_ops will hang indefinitely
        # for faster dequeuing the batch size should be fixed, to be able to use q.dequeue_many(qsize)
        before = time.time()
        all_sizes = empty_all_queues(sess, self.queues_size, self.queues_dequeue)
        after = time.time()
        diff = after-before
        tf_logging.info("dequeuing time is %s:%s"%(str(int(diff//60)), str(diff%60).zfill(2)))

        # naive dequeuing (slow)
        # all_sizes = []
        # for qs, qd in zip(:
        #     qsize = sess.run(dq["size"])
        #     all_sizes.append(dq["size"])
        #     for i in range(qsize):
        #         sess.run(dq["dequeue"])

        assert all(all_sizes==0), "queues not succefully emptied after decoding"

        return audio_batch, np.array(losses)

    def mean_reconstr_loss(self, individual_losses):
        '''This method is used in WavGenerateHook in order to be able to compute the loss separately for hs, zs train
            and validation. This is because we decode all of the above samples at once to reduce time
        '''
        return np.mean(individual_losses)

    def decode_debug_astf(self, Z, X_shifted, sess=None):
        """Decode latent vectors in input data for wavenet autoregressive model,
        test of the fast generation to see if it is equal to the tf. It should be."""
        sess = sess or self.get_raw_session()

        bs, z_length, z_ch = Z.shape
        hop_length = self._network._hop_length
        time_length = z_length * hop_length

        assert X_shifted.shape == (bs, time_length, 1)

        # initialize the fast generation queues with all 0s, x_t is needed for the batch_size
        sess.run(self.queues_init_ops, feed_dict={self.x_t_qs : X_shifted[:,:1,:]})

        audio_batch = np.zeros((bs, time_length, 1), dtype=np.float32)

        before = time.time()
        for t in range(time_length):
            i_z = t // hop_length
            x = sess.run([self.x_tp1, self.queues_push_ops],
                feed_dict={self.x_t_qs : X_shifted[:,t:t+1,:], self.z_t : Z[:, i_z:i_z+1, :]})[0]
            audio_batch[:, t, 0] = x[:, 0, 0]
            if t % 1000 == 0:
                tf_logging.info("Sample: %d" % t)
        after = time.time()
        diff = after-before
        tf_logging.info("decoding time is %s:%s"%(str(int(diff//60)), str(diff%60).zfill(2)))

        #NB(Riccardo) I dequeue all the elements from the queues, otherwise next call to queues_init_ops will hang indefinitely
        # for faster dequeuing the batch size should be fixed, to be able to use q.dequeue_many(qsize)

        # smart dequeuing contemporaneously on all queues till I can
        before = time.time()
        all_sizes = empty_all_queues(sess, self.queues_size, self.queues_dequeue)
        after = time.time()
        diff = after-before
        tf_logging.info("dequeuing time is %s:%s"%(str(int(diff//60)), str(diff%60).zfill(2)))

        assert all(all_sizes==0), "queues not succefully emptied after decoding"

        return audio_batch

    def reconstruct(self, X, sess=None):
        """ Use AE to reconstruct given data.
        With teacher forcing.
        """
        sess = sess or self.get_raw_session()

        Z, _ = self.encode(X, sess=sess)

        return self.decode(Z, sess=sess)

    def generate(self, X):
        pass

