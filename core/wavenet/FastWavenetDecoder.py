import sonnet as snt
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .utils import condition, inv_mu_law, mu_law

# def generate_audio_sample(sess, net, audio, encoding):
#     """Generate a single sample of audio from an encoding.
#
#     Args:
#     sess: tf.Session to use.
#     net: Loaded wavenet network (dictionary of endpoint tensors).
#     audio: Previously generated audio [batch_size, 1].
#     encoding: Encoding at current time index [batch_size, dim].
#
#     Returns:
#     audio_gen: Generated audio [batch_size, 1]
#     """
#     probability_mass_function = sess.run(
#       [net["predictions"], net["push_ops"]],
#       feed_dict={net["X"]: audio, net["encoding"]: encoding})[0]
#     #TODO sample_bin is a sample from the softmax, use tf distribution for this...
#     sample_bin = sample_categorical(probability_mass_function)
#     audio_gen = utils.inv_mu_law_numpy(sample_bin - 128)
#     return audio_gen
#
#
# def load_fastgen_nsynth(batch_size=1):
#   """Load the NSynth fast generation network.
#
#   Args:
#     batch_size: Batch size number of observations to process. [1]
#   Returns:
#     graph: The network as a dict with input placeholder in {"X"}
#   """
#   config = FastGenerationConfig(batch_size=batch_size)
#   with tf.device("/gpu:0"):
#     x = tf.placeholder(tf.float32, shape=[batch_size, 1])
#     graph = config.build({"wav": x})
#     graph.update({"X": x})
#   return graph
#
# #
# # def sample_categorical(probability_mass_function):
# #   """Sample from a categorical distribution.
# #
# #   Args:
# #     probability_mass_function: Output of a softmax over categories.
# #       Array of shape [batch_size, number of categories]. Rows sum to 1.
# #
# #   Returns:
# #     idxs: Array of size [batch_size, 1]. Integer of category sampled.
# #   """
# #   if probability_mass_function.ndim == 1:
# #     probability_mass_function = np.expand_dims(probability_mass_function, 0)
# #   batch_size = probability_mass_function.shape[0]
# #   cumulative_density_function = np.cumsum(probability_mass_function, axis=1)
# #   rand_vals = np.random.rand(batch_size)
# #   idxs = np.zeros([batch_size, 1])
# #   for i in range(batch_size):
# #     idxs[i] = cumulative_density_function[i].searchsorted(rand_vals[i])
# #   return idxs
# #
# #
# def synthesize(encodings,
#                save_paths,
#                checkpoint_path="model.ckpt-200000",
#                samples_per_save=10000):
#   """Synthesize audio from an array of encodings.
#
#   Args:
#     encodings: Numpy array with shape [batch_size, time, dim].
#     save_paths: Iterable of output file names.
#     checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
#     samples_per_save: Save files after every amount of generated samples.
#   """
#   session_config = tf.ConfigProto(allow_soft_placement=True)
#   session_config.gpu_options.allow_growth = True
#   with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
#     net = load_fastgen_nsynth(batch_size=encodings.shape[0])
#     saver = tf.train.Saver()
#     saver.restore(sess, checkpoint_path)
#
#     # Get lengths
#     batch_size, encoding_length, _ = encodings.shape
#     hop_length = Config().ae_hop_length
#     total_length = encoding_length * hop_length
#
#     # initialize queues w/ 0s
#     sess.run(net["init_ops"])
#
#     # Regenerate the audio file sample by sample
#     audio_batch = np.zeros(
#         (batch_size, total_length), dtype=np.float32)
#     audio = np.zeros([batch_size, 1])
#
#     for sample_i in range(total_length):
#       encoding_i = sample_i // hop_length
#       audio = generate_audio_sample(sess, net,
#                                     audio, encodings[:, encoding_i, :])
#       audio_batch[:, sample_i] = audio[:, 0]
#       if sample_i % 100 == 0:
#         tf.logging.info("Sample: %d" % sample_i)
#       if sample_i % samples_per_save == 0 and save_paths:
#         save_batch(audio_batch, save_paths)
#
#     save_batch(audio_batch, save_paths)
#


class FastWavenetDecoder(snt.AbstractModule):
    """The FastWaveNetDecoder
        it takes a single value in time, assumes x = [mb, 1, ch]
        and it will output a single prediction for the next value of x
    """

    def __init__(self, num_layers_per_stage,
                 num_layers,
                 filter_length,
                 d_hidden_channels,
                 skip_channels,
                 latent_channels,
                 regularizers={},
                 variable_scope, name='fast_wavenet_decoder'):
        super(FastWavenetDecoder, self).__init__(name=name)
        self._num_layers_per_stage = num_layers_per_stage
        self._num_layers = num_layers
        self._filter_length = filter_length
        self._d_hidden_channels = d_hidden_channels
        self._skip_channels = skip_channels
        self._latent_channels = latent_channels
        self._regularizers = regularizers
        self._other_scope_name = variable_scope

    def _build(self, x_t, z_t, conditioning_t=None):
        """

        Args:
            x_t: it is the previous generation at a particular time, shape = [bs, 1, ch]
            z_t: it is the encoding at a particular time, shape = [bs, 1, latent_channels]
            conditioning_t: the conditioning at a particular time.
        Returns:

        """

        bs = tf.shape(x_t)[0]

        x_channels = x_t.shape[2]

        self.queues_dicts = []
        self._nodes_list = []

        l, queues_dict = self._causal_conv_queued(x_t,
                                                   batch_size=bs,
                                                   ch_in=x_channels,
                                                   ch_out=self._d_hidden_channels,
                                                   filter_length=self._filter_length,
                                                   dilation=1,
                                                   name="startconv")

        for qd in queues_dict:
            self.queues_dicts.append(qd)

        self._nodes_list.append(l)

        # Set up skip connections.
        s = self._linear(l,
                         ch_in=self._d_hidden_channels,
                         ch_out=self._skip_channels,
                         name="skip_start")

        self._nodes_list.append(s)

        # Residual blocks with skip connections.
        for i in range(self._num_layers):
            dilation = 2 ** (i % self._num_layers_per_stage)

            d, queues_dict = self._causal_conv_queued(l,
                                       batch_size=bs,
                                       ch_in=self._d_hidden_channels,
                                       ch_out=self._d_hidden_channels*2,
                                       filter_length=self._filter_length,
                                       dilation=dilation,
                                       name= "dilatedconv_%d" % (i + 1))

            for qd in queues_dict:
                self.queues_dicts.append(qd)

            self._nodes_list.append(d)

            # local conditioning for this specific time
            d += self._linear(z_t,
                            ch_in=self._latent_channels,
                            ch_out=self._d_hidden_channels * 2,
                            name='cond_map_%d' % (i + 1))

            self._nodes_list.append(d)

            # gated cnn
            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d = tf.sigmoid(d[:, :, :m]) * tf.tanh(d[:, :, m:])

            # residual connection
            l += self._linear(d,
                            ch_in=self._d_hidden_channels,
                            ch_out=self._d_hidden_channels,
                            name='res_%d' % (i + 1))

            self._nodes_list.append(l)

            # skip connections
            s += self._linear(d,
                            ch_in=self._d_hidden_channels,
                            ch_out=self._skip_channels,
                            name='skip_%d' % (i + 1))

            self._nodes_list.append(s)

        s = tf.nn.relu(s)

        s = self._linear(s,
                     ch_in=self._skip_channels,
                     ch_out=self._skip_channels,
                     name='out1')

        self._nodes_list.append(s)

        #local conditioning for this specific time
        s += self._linear(z_t,
                ch_in = self._latent_channels,
                ch_out = self._skip_channels,
                name='cond_map_out1')

        s = tf.nn.relu(s)

        self._nodes_list.append(s)

        self.logits = self._linear(s,
                            ch_in=self._skip_channels,
                            ch_out=256,
                            name='logits')

        # x_{t+1}
        self.x_tp1_distr = tfd.Categorical(logits=self.logits)
        # dimension of x_{t+1} should be: [bs, 1, 1]
        self.x_tp1 = inv_mu_law(tf.expand_dims(self.x_tp1_distr.mode(), axis=-1) - 128, name="x_tp1")

        return self.x_tp1_distr, self.x_tp1, self.queues_dicts

    def _get_layer_vars(self, layer_name):
        layer_scope_name = self._other_scope_name + "/" + layer_name

        ws = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = layer_scope_name + "/w")
        if not ws:
            raise Exception("found no weights in scope: %s"%layer_scope_name)
        elif len(ws)>1:
            raise Exception("found more than one weight in scope: %s"
                            "\n%s"%(layer_scope_name, str(ws)))
        w = ws[0]

        bs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = layer_scope_name + "/b")
        if not bs:
            raise Exception("found no biases in scope: %s"%layer_scope_name)
        elif len(bs)>1:
            raise Exception("found more than one bias in scope: %s"
                            "\n%s" % (layer_scope_name, str(bs)))
        b = bs[0]

        return w,b

    def _check_shape(self, var, var_shape, name):
        assert var_shape == var.shape, "variable dimension `%s` did not match with assumed shape `%s` " \
                                            "for layer %s" % (str(var.shape), str(var_shape), name)

    def _linear(self, x, ch_in, ch_out, name):
        """Simple linear layer.

        Args:
          x: The [mb, 1, ch_in] tensor input.
          ch_in: The input number of channels.
          ch_out: The output number of channels.
          name: The variable scope where to find the weights.

        Returns:
          y: The output of the operation.
        """

        w, b = self._get_layer_vars(name)

        w_shape = [1, ch_in, ch_out]
        self._check_shape(w, w_shape, name)

        b_shape = [ch_out]
        self._check_shape(b, b_shape, name)

        y = tf.nn.bias_add(tf.matmul(x[:, 0, :], w[0]), b)
        y = tf.expand_dims(y, 1)
        return y


    def _causal_conv_queued(self, x, batch_size, ch_in, ch_out, filter_length, dilation, name):
        """Applies dilated convolution using queues.

        Assumes a filter_length of 3.

        Args:
        x: The [bs, 1, ch_in] tensor input.
        batch_size : the batch size, needed to init the queues
        ch_in: The input number of channels.
        ch_out: The output number of channels.
        filter_length: The length of the convolution, assumed to be 3.
        rate: The rate or dilation
        batch_size: Non-symbolic value for batch_size.
        name: The variable scope where to find the weights.

        Returns:
        y: The output of the operation
        (init_1, init_2): Initialization operations for the queues
        (push_1, push_2): Push operations for the queues
        """
        assert filter_length == 3

        # x is passed here with shape [bs, 1, ch_in]
        x_shape = [None, 1, ch_in]

        # create queues, one for each 'leg' of the dilated convolution (besides the direct connection from the bottom)
        q_1 = tf.FIFOQueue(dilation, dtypes=[tf.float32]) #, shapes=[[None, 1, ch_in]])
        q_2 = tf.FIFOQueue(dilation, dtypes=[tf.float32]) #, shapes=[[None, 1, ch_in]])

        init_1 = q_1.enqueue_many(tf.zeros([dilation, batch_size, 1, ch_in]))
        init_2 = q_2.enqueue_many(tf.zeros([dilation, batch_size, 1, ch_in]))

        size_1 = q_1.size()
        size_2 = q_2.size()

        state_1 = q_1.dequeue()
        state_1.set_shape(x_shape)
        push_1 = q_1.enqueue(x)
        state_2 = q_2.dequeue()
        state_2.set_shape(x_shape)
        push_2 = q_2.enqueue(state_1)

        # get pretrained weights
        w, b = self._get_layer_vars(name)

        w_shape = [filter_length, ch_in, ch_out]
        self._check_shape(w, w_shape, name)

        b_shape = [ch_out]
        self._check_shape(b, b_shape, name)

        w_q_2 = tf.slice(w, [0, 0, 0], [1, -1, -1])
        w_q_1 = tf.slice(w, [1, 0, 0], [1, -1, -1])
        w_x = tf.slice(w, [2, 0, 0], [1, -1, -1])

        # perform op w/ cached states
        y = tf.nn.bias_add(
          tf.matmul(state_2[:, 0, :], w_q_2[0]) + tf.matmul(
              state_1[:, 0, :], w_q_1[0]) + tf.matmul(x[:, 0, :], w_x[0]), b)

        y = tf.expand_dims(y, 1)

        return y, ({ "init" : init_1,
                     "push" : push_1,
                     "dequeue" : state_1,
                     "size" : size_1},
                   {"init": init_2,
                    "push": push_2,
                    "dequeue": state_2,
                    "size": size_2})

