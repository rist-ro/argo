import tensorflow as tf

from argo.core.network.ArgoAbstractNetwork import ArgoAbstractNetwork
from .wavenet.AlmostWavenetEncoder import AlmostWavenetEncoder
from .wavenet.FastWavenetDecoder import FastWavenetDecoder
from .wavenet.WavenetDecoderTeacherForcing import WavenetDecoderTeacherForcing
from .wavenet.utils import shift_right, mu_law, pool1d

DIM_REDUCTION_CONV = 'conv'

DIM_REDUCTION_MAX_POOL = 'max_pool'

DIM_REDUCTION_AVG_POOL = 'avg_pool'

DIM_REDUCTION_LINEAR = 'linear'

class WavenetAENetwork(ArgoAbstractNetwork):
    """
    Network class for managing a WavenetAENetwork
    """

    # if not present in opts, the values specified in default_params will be provided automagically
    # see more in AbstractNetwork.__init__
    default_params = {
        **ArgoAbstractNetwork.default_params,
        "network_architecture": {
            "num_layers_per_stage": 10,
            "num_layers": 30,
            "filter_length": 3,
            "d_hidden_channels": 512,
            "hop_length": 512,
            "e_hidden_channels": 128,
            "skip_channels": 256,
            "latent_channels": 16,
            "dim_reduction": "avg_pool",
            "p_dropout_decoder_tf": 0,
            "upsample_encoding": False
        },
    }

    def create_id(self):
        opts = self._opts
        _id = '-st' + str(opts["network_architecture"]["num_layers_per_stage"])
        _id += '-l' + str(opts["network_architecture"]["num_layers"])
        _id += '-fl' + str(opts["network_architecture"]["filter_length"])
        _id += '-hc' + str(opts["network_architecture"]["d_hidden_channels"])
        _id += '-hl' + str(opts["network_architecture"]["hop_length"])
        _id += '-oc' + str(opts["network_architecture"]["e_hidden_channels"])
        _id += '-sc' + str(opts["network_architecture"]["skip_channels"])
        _id += '-lc' + str(opts["network_architecture"]["latent_channels"])
        _id += '-dr_' + opts['network_architecture']['dim_reduction']
        _id += '-dp' + str(opts['network_architecture']['p_dropout_decoder_tf'])
        _id += '-up' + str(int(opts['network_architecture']['upsample_encoding']))

        super_id = super().create_id()

        _id += super_id
        return _id

    def __init__(self, opts, name="wavenet_ae_network"):
        """Short summary.

        Args:
            opts (dict): parameters of the task.
            name (str): name of the Sonnet module.
        """
        super().__init__(opts, name, opts["seed"])
        network_architecture = self._opts["network_architecture"]

        self._num_layers_per_stage = network_architecture["num_layers_per_stage"]
        self._num_layers = network_architecture["num_layers"]
        self._filter_length = network_architecture["filter_length"]  # n_channels of h
        self._d_hidden_channels = network_architecture["d_hidden_channels"]  # n_channels of z
        self._hop_length = network_architecture["hop_length"]  # n_channels of z
        self._e_hidden_channels = network_architecture["e_hidden_channels"]  # n_samples
        self._skip_channels = network_architecture["skip_channels"]  # n_samples
        self._latent_channels = network_architecture["latent_channels"]  # n_samples
        self._dim_reduction = network_architecture['dim_reduction']  # type of dimensionality reduction to use
        self._prob_dropout_decoder_tf = network_architecture['p_dropout_decoder_tf']  # prob to dropout inputs from decoder with tf
        self._upsample_encoding = network_architecture['upsample_encoding']
        

    def _build(self, x, network_str=""):
        """
        Args:
            x (tf.tensor): input node.
            network_str (str): Optional network_str specifying the network we are going to build.
                            It is used to set some specific collections for activity and contractive regularizers.

        Returns:
            tf distribution on the visible neurons
            tf distribution of the latent space

        """

        self._encoder = AlmostWavenetEncoder(self._num_layers_per_stage,
                                               self._num_layers,
                                               self._filter_length,
                                               self._hop_length,
                                               self._e_hidden_channels,
                                               self._latent_channels,
                                             name = 'enc')

        self._decoder_teacherforcing = WavenetDecoderTeacherForcing(self._num_layers_per_stage,
                                                                    self._num_layers,
                                                                    self._filter_length,
                                                                    self._d_hidden_channels,
                                                                    self._skip_channels,
                                                                    self._prob_dropout_decoder_tf,
                                                                    name = 'dec_tf')

        self._decoder = FastWavenetDecoder(self._num_layers_per_stage,
                                           self._num_layers,
                                           self._filter_length,
                                           self._d_hidden_channels,
                                           self._skip_channels,
                                           self._latent_channels,
                                           variable_scope = self._decoder_teacherforcing.scope_name,
                                           name = 'dec')

        # Encode the source with 8-bit Mu-Law.
        # mu_law takes input [-1, 1] and encodes in mu(=256) discrete values
        # preprocessing output x_quant_scaled, is quantized between -1 and 1
        with tf.variable_scope("quantize"):
            self.x_quantized = mu_law(x)
            # x_quant_scaled is quantized in [-1, 1]
            self.x_quant_scaled = tf.cast(self.x_quantized, tf.float32) / 128.0

        self.en = self._encoder(self.x_quant_scaled)

        # self.z = pool1d(self.en, self._hop_length, name='ae_pool', mode='avg')
        self.z = self.reduce_dimension_layer(self.en)

        self.linear_upsample_layer = tf.layers.Dense(units=self._hop_length)
        z_transpose = tf.transpose(self.z, perm=[0, 2, 1])  # (bs, ch, time/hoplen)
        z_transpose = tf.expand_dims(z_transpose, axis=-1)
        z_upsampled = self.linear_upsample_layer(z_transpose)  # (bs, ch, time/hoplen, hoplen)
        self.z_upsampled = tf.reshape(z_upsampled, [tf.shape(z_upsampled)[0], -1, self._latent_channels])

        self.x_shifted_qs = shift_right(self.x_quant_scaled)  # loose the last x, and pad at the beginning with a 0

        z = self.z_upsampled if self._upsample_encoding else self.z
        self.x_reconstruction_distr_tf, self.x_reconstruction_node_tf = self._decoder_teacherforcing(self.x_shifted_qs, z)

        # assume 1 channel audio
        self.x_t = tf.placeholder(tf.float32, [None, 1, 1], name="x_t")

        # Encode the source with 8-bit Mu-Law.
        # mu_law takes input [-1, 1] and encodes in mu(=256) discrete values
        # preprocessing output x_t_quant_scaled, is quantized between -1 and 1
        with tf.variable_scope("quantize_t"):
            self.x_t_quantized = mu_law(self.x_t)
            # x_t_quant_scaled is quantized in [-1, 1]
            self.x_t_quant_scaled = tf.cast(self.x_t_quantized, tf.float32) / 128.0

        self.z_t = tf.placeholder(tf.float32, [None, 1, self._latent_channels], name="z_t")

        self.x_tp1_distr, self.x_tp1, queues_dicts = self._decoder(self.x_t_quant_scaled, self.z_t)

        self.queues_init_ops = [qd["init"] for qd in queues_dicts]
        self.queues_push_ops = [qd["push"] for qd in queues_dicts]

        self.queues_dicts = queues_dicts

        return {
                "x_shifted_qs" : self.x_shifted_qs,
                "z" : self.z,
                "z_upsampled": self.z_upsampled,
                "upsample_encoding": self._upsample_encoding,
                "x_rec_distr_tf" : self.x_reconstruction_distr_tf,
                "x_rec_tf" : self.x_reconstruction_node_tf,
                "x_t" : self.x_t,
                "x_t_qs": self.x_t_quant_scaled,
                "z_t" : self.z_t,
                "x_tp1_distr" : self.x_tp1_distr,
                "x_tp1" : self.x_tp1,
                "queues_init_ops" : self.queues_init_ops,
                "queues_push_ops" : self.queues_push_ops,
                "queues_dicts" : self.queues_dicts,
            }

    def reduce_dimension_layer(self, inputs):
        '''
        reduces the inputs based on the self._dim_reduction constant
        Args:
            inputs (tf.Tensor): inputs of shape (batch_size, signal_length, latent_channels)

        Returns:
            tf.Tensor: reduced inputs based on self._hop_length
                       with shape (batch_size, signal_length // self._hop_length, latent_channels)
        '''
        if self._dim_reduction == DIM_REDUCTION_MAX_POOL:
            reduced = pool1d(inputs, self._hop_length, name='ae_pool', mode='max')

        elif self._dim_reduction == DIM_REDUCTION_AVG_POOL:
            reduced = pool1d(inputs, self._hop_length, name='ae_pool', mode='avg')

        elif self._dim_reduction == DIM_REDUCTION_CONV:
            reduced = tf.layers.Conv1D(filters=self._latent_channels,
                                    kernel_size=self._hop_length,
                                    strides=self._hop_length,
                                    padding='same')(inputs)

        elif self._dim_reduction == DIM_REDUCTION_LINEAR:
            # because Dense can only be applied on last dimension we have to change the order of dimensions
            reduced = tf.transpose(inputs, perm=[0, 2, 1])
            # only transposing dimensions won;t help because None shape dimensions so we have to reshape
            # (bs, channels, signal//hop_len, hop_length) and apply linear layer of 1 unit to get
            # -> (bs, channels, signal//hop_len, 1) kudos @Csongor
            reduced_reshaped = tf.reshape(reduced, [tf.shape(inputs)[0], inputs.shape[-1], -1, self._hop_length])
            reduced = tf.layers.Dense(units=1)(reduced_reshaped)
            reduced = tf.squeeze(reduced, axis=-1)
            reduced = tf.transpose(reduced, perm=[0, 2, 1])

        else:
            raise ValueError('Can\'t recognize type of dimensionality reduction method: \'{}\' '
                             '(change network architecture -> dim_reduction: <max_pool | avg_pool | conv | linear>'
                             .format(self._dim_reduction)
                             )

        return reduced

# THIS IS NAIVE GENERATIVE NETWORK BUT IT IS SUPER SLOW o:
# decoder_module = self._decoder
# ch = x.shape[2]
#
# def _build_gen_net(self, decoder_module, z_node, ch):
#
#     def body(i, x_in, length, z):
#         distr, x_rec = decoder_module(x_in, z)
#         last_x_rec = tf.slice(x_rec, [0, length-1, 0], [-1, 1, ch])
#         #concat on the temporal dimension
#         x_concat = tf.concat([last_x_rec, x_in], axis=1)
#         x_out = tf.slice(x_concat, [0, 0, 0], tf.stack([-1, length, ch]))
#         i+=1
#         return (i, x_out, length, z)
#
#     def cond(i, x_gen, length, z):
#         return i<length
#
#     gen_batch = tf.shape(z_node)[0]
#     gen_length = tf.shape(z_node)[1] * self._hop_length
#     gen_shape = [gen_batch, gen_length, ch]
#     x_gen = tf.zeros(gen_shape)
#     i = 0
#
#     loop_vars = [i, x_gen, gen_length, z_node]
#     i, x_gen, gen_length, z = tf.while_loop(cond, body, loop_vars)
#
#     return x_gen
