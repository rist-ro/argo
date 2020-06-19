import sonnet as snt
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from argo.core.network.ArgoStochasticNetworkWithDefaults import ArgoStochasticNetworkWithDefaults
from argo.core.network.ArgoAbstractNetwork import ArgoAbstractNetwork
from .wavenet.GaussianDiagonal import GaussianDiagonal
from .wavenet.GaussianTriDiagonal import GaussianTriDiagonal
from .wavenet.GaussianTriDiagonalPrecision import GaussianTriDiagonalPrecision
from .wavenet.IndependentChannelsTransformedDistribution import IndependentChannelsTransformedDistribution
from .wavenet.AlmostWavenetEncoder import AlmostWavenetEncoder
from .wavenet.FastWavenetDecoder import FastWavenetDecoder
from .wavenet.WavenetDecoderTeacherForcing import WavenetDecoderTeacherForcing
from .wavenet.utils import shift_right, mu_law, pool1d

GAUSSIAN_TRI_DIAGONAL_PRECISION = "GaussianTriDiagonalPrecision"

GAUSSIAN_TRI_DIAGONAL = "GaussianTriDiagonal"

GAUSSIAN_DIAGONAL = "GaussianDiagonal"

DIM_REDUCTION_CONV = 'conv'

DIM_REDUCTION_MAX_POOL = 'max_pool'

DIM_REDUCTION_AVG_POOL = 'avg_pool'

DIM_REDUCTION_LINEAR = 'linear'

LEARN_PRIOR_MAF = 'MAF'

LEARN_PRIOR_CONV = 'CONV'

LEARN_PRIOR_NO = 'NO'


class WavenetVAENetwork(ArgoStochasticNetworkWithDefaults):
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
            "variational_layer": "GaussianDiagonal",
            "learn_prior": LEARN_PRIOR_NO,
            "upsample_encoding": True,
            "flow_params": {'name': 'MAF', 'num_bijectors': 3, 'hidden_channels': 512, 'permute': False},
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
        _id += '-ar' + str(opts["network_architecture"]["alpha_rescale"])
        _id += '-dr_' + opts['network_architecture']['dim_reduction']
        _id += '-dp' + str(opts['network_architecture']['p_dropout_decoder_tf'])
        _id += '-lp' + str(opts['network_architecture']['learn_prior'])
        _id += '-up' + str(int(opts['network_architecture']['upsample_encoding']))

        latent = ""
        if opts["network_architecture"]["variational_layer"] == "GaussianDiagonal":
            latent = "GD"
        elif opts["network_architecture"]["variational_layer"] == "GaussianTriDiagonal":
            latent = "GTD"
        elif opts["network_architecture"]["variational_layer"] == "GaussianTriDiagonalPrecision":
            latent = "GTDP"
        else:
            raise ValueError("Id for {} not defined...".format(opts["network_architecture"]["variational_layer"]))

        _id += '-vl' + latent

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
        self._variational_layer = network_architecture["variational_layer"]  # n_samples
        self.n_z_samples = network_architecture["n_z_samples"]
        self._alpha = network_architecture["alpha_rescale"]
        self._dim_reduction = network_architecture['dim_reduction']  # type of dimensionality reduction to use
        self._prob_dropout_decoder_tf = network_architecture['p_dropout_decoder_tf']  # prob to dropout inputs from decoder with tf
        self._learn_prior = network_architecture['learn_prior']
        self._upsample_encoding = network_architecture['upsample_encoding']
        self._flow_params = network_architecture["flow_params"]

    def _build(self, x, x_shape, network_str="", conditioning=None):
        """
        Args:
            x (tf.tensor): input node.
            network_str (str): Optional network_str specifying the network we are going to build.
                            It is used to set some specific collections for activity and contractive regularizers.

        Returns:
            tf distribution on the visible neurons
            tf distribution of the latent space

        """
        self._x_shape = x_shape
        self.time_dim_latent = self._x_shape[0] // self._hop_length
        # x.set_shape([None, *x_shape])

        self._encoder = AlmostWavenetEncoder(self._num_layers_per_stage,
                                             self._num_layers,
                                             self._filter_length,
                                             self._hop_length,
                                             self._e_hidden_channels,
                                             self._latent_channels,
                                             self._de
                                             name='enc')

        self._decoder_teacherforcing = WavenetDecoderTeacherForcing(self._num_layers_per_stage,
                                                                    self._num_layers,
                                                                    self._filter_length,
                                                                    self._d_hidden_channels,
                                                                    self._skip_channels,
                                                                    self._prob_dropout_decoder_tf,
                                                                    name='dec_tf')

        self._decoder = FastWavenetDecoder(self._num_layers_per_stage,
                                           self._num_layers,
                                           self._filter_length,
                                           self._d_hidden_channels,
                                           self._skip_channels,
                                           self._latent_channels,
                                           regularizers={"w": self._default_weights_reg, "b": self._default_bias_reg}
                                           variable_scope=self._decoder_teacherforcing.scope_name,
                                           name='dec')

        # Encode the source with 8-bit Mu-Law.
        # mu_law takes input [-1, 1] and encodes in mu(=256) discrete values
        # preprocessing output x_quant_scaled, is quantized between -1 and 1
        with tf.variable_scope("quantize"):
            self.x_quantized = mu_law(x)
            # x_quant_scaled is quantized in [-1, 1]
            self.x_quant_scaled = tf.cast(self.x_quantized, tf.float32) / 128.0

        self.en = self._encoder(self.x_quant_scaled)
        self.en = self.reduce_dimension_layer(self.en)

        self.variational_layer()

        with tf.variable_scope('latent_sampling'):
            batch_size = self._gaussian_model_latent.batch_shape_tensor()[0]
            time_size = self._gaussian_model_latent.event_shape_tensor()[0]
            ch_size = self.en.shape[2]
            # ch_size = self._gaussian_model_latent.batch_shape.as_list()[2]
            batch_reshape = [self.n_z_samples * batch_size, self._latent_channels, self.time_dim_latent]

            self.samples_posterior = self._gaussian_model_latent.sample(self.n_z_samples)

            sample_shape = self.samples_posterior.get_shape().as_list()
            assert sample_shape[-2:] == batch_reshape[-2:], 'Posterior sample does not have right shape'

            self.samples_posterior = tf.reshape(self.samples_posterior, batch_reshape)
            self.z = tf.transpose(self.samples_posterior, perm=[0, 2, 1]) #  from (n_samples * bs, ch, time) -> (n_samples * bs, time, ch)
            self.log_pdf_z = self._gaussian_model_latent.log_prob(self.samples_posterior)
            # self.z = tf.reshape(self._gaussian_model_latent.sample(self.n_z_samples), batch_reshape)
            # self.log_pdf_z = self._gaussian_model_latent.log_prob(tf.transpose(self.z, perm=[0, 2, 1]))

        self.x_shifted_qs = shift_right(self.x_quant_scaled)

        self.linear_upsample_layer = tf.layers.Dense(units=self._hop_length, name='upsample_encoding')
        z_transpose = tf.transpose(self.z, perm=[0, 2, 1])  # (bs, ch, time/hoplen)
        z_transpose = tf.expand_dims(z_transpose, axis=-1)
        z_upsampled = self.linear_upsample_layer(z_transpose)  # (bs, ch, time/hoplen, hoplen)
        self.z_upsampled = tf.reshape(z_upsampled, [tf.shape(z_upsampled)[0], -1, self._latent_channels])  # (bs, time, ch)

        # self.z_upsampled = tf.Print(self.z_upsampled, [tf.shape(self.z_upsampled)], '\n\n\nShape of z upsampled\n\n\n')

        self.teacher_forcing_decoder(conditioning)

        self.fast_generation_decoder(conditioning)

        return {
            "x_shifted_qs":    self.x_shifted_qs,
            "z":               self.z,
            "samples_posterior": self.samples_posterior,
            "z_upsampled":     self.z_upsampled,
            "upsample_encoding": self._upsample_encoding,
            "latent":          self._gaussian_model_latent,
            "x_rec_distr_tf":  self.x_reconstruction_distr_tf,
            "x_rec_tf":        self.x_reconstruction_node_tf,
            "x_t":             self.x_t,
            "x_t_qs":          self.x_t_quant_scaled,
            "z_t":             self.z_t,
            "x_tp1_distr":     self.x_tp1_distr,
            "x_tp1":           self.x_tp1,
            "queues_init_ops": self.queues_init_ops,
            "queues_push_ops": self.queues_push_ops,
            "queues_dicts":    self.queues_dicts,
            "n_z_samples":     self.n_z_samples,
            "prior":           self._prior,
        }

    def teacher_forcing_decoder(self, conditioning=None):  # loose the last x, and pad at the beginning with a 0
        z = self.z_upsampled if self._upsample_encoding else self.z
        input_shape = self.x_shifted_qs.shape.as_list()[1:]
        ones = [1] * len(input_shape)
        x_shifted_replicate = tf.tile(self.x_shifted_qs, [self.n_z_samples] + ones)

        self.x_reconstruction_distr_tf, self.x_reconstruction_node_tf = self._decoder_teacherforcing(x_shifted_replicate,
                                                                                                     z,
                                                                                                     conditioning)

    def fast_generation_decoder(self, conditioning=None):
        # assume 1 channel audio
        self.x_t = tf.placeholder(tf.float32, [None, 1, 1], name="x_t")

        self.z_t = tf.placeholder(tf.float32, [None, 1, self._latent_channels], name="z_t")

        # Encode the source with 8-bit Mu-Law.
        # mu_law takes input [-1, 1] and encodes in mu(=256) discrete values
        # preprocessing output x_t_quant_scaled, is quantized between -1 and 1
        with tf.variable_scope("quantize_t"):
            self.x_t_quantized = mu_law(self.x_t)
            # x_t_quant_scaled is quantized in [-1, 1]
            self.x_t_quant_scaled = tf.cast(self.x_t_quantized, tf.float32) / 128.0

        self.x_tp1_distr, self.x_tp1, queues_dicts = self._decoder(self.x_t_quant_scaled, self.z_t, conditioning)

        self.queues_init_ops = [qd["init"] for qd in queues_dicts]
        self.queues_push_ops = [qd["push"] for qd in queues_dicts]

        self.queues_dicts = queues_dicts

    def variational_layer(self):
        if self._learn_prior == LEARN_PRIOR_CONV:
            self.en_shifted = shift_right(self.en)

            # POSTERIOR
            self.en_posterior = snt.Conv1D(output_channels=self._latent_channels,
                                           kernel_shape=1,
                                           stride=1,
                                           padding=snt.CAUSAL,
                                           name='conv_en_posterior')(self.en_shifted)

            self.input_conv = snt.Conv1D(output_channels=self._latent_channels,
                                         kernel_shape=self._hop_length,
                                         stride=self._hop_length,
                                         padding=snt.CAUSAL,
                                         name='conv_en_input')(self.x_quant_scaled)

            self.en_posterior = tf.concat([self.en_posterior, self.input_conv], axis=-1)
            self._gaussian_model_latent = self.get_gaussian(self.en_posterior)

            # PRIOR
            self.en_prior = snt.Conv1D(output_channels=self._latent_channels,
                                       kernel_shape=1,
                                       stride=1,
                                       padding=snt.CAUSAL,
                                       name='conv_en_prior_from_input')(shift_right(self.x_quant_scaled))

            self.en_prior = pool1d(self.en_prior, self._hop_length, name='ae_pool_prior', mode='avg')

            self._prior = self.get_gaussian(self.en_prior)

        else:
            self._gaussian_model_latent = self.get_gaussian(self.en)
            self._create_prior()

    def _create_prior(self):
        if self._learn_prior == LEARN_PRIOR_MAF:
            self._prior = IndependentChannelsTransformedDistribution(
                [1, self.time_dim_latent], self._flow_params, self.time_dim_latent, self._latent_channels
            )

        elif self._variational_layer == GAUSSIAN_DIAGONAL:
            dim = tf.shape(self._gaussian_model_latent.mean())[1:3]
            zeros = tf.zeros(shape=dim)
            ones = self._alpha * tf.ones(shape=dim)
            self._prior = tfd.MultivariateNormalDiag(loc=zeros, scale_diag=ones, name="prior")

        elif self._variational_layer == GAUSSIAN_TRI_DIAGONAL:
            dim = tf.shape(self._gaussian_model_latent.mean())[1:3]
            ch_z = self._gaussian_model_latent.mean().shape[1]
            zeros = tf.zeros(shape=dim)
            ones = tf.stack([tf.eye(dim[1]) for i in range(ch_z)])
            twos = self._alpha * 0.2 * tf.stack([tf.eye(dim[1] - 1) for i in range(ch_z)])
            twos_big = tf.pad(twos, [[0, 0], [1, 0], [0, 1]], mode='CONSTANT')
            cov = ones + twos_big + tf.transpose(twos_big, perm=[0, 2, 1])
            self._prior = tfd.MultivariateNormalFullCovariance(loc=zeros, covariance_matrix=cov, name="prior")

        elif self._variational_layer == GAUSSIAN_TRI_DIAGONAL_PRECISION:
            dim = tf.shape(self._gaussian_model_latent.mean())[1:3]
            ch_z = self._gaussian_model_latent.mean().shape[1]
            zeros = tf.zeros(shape=dim)

            ones = tf.stack([tf.eye(dim[1]) for i in range(ch_z)])

            twos = self._alpha * tf.stack([tf.eye(dim[1] - 1) for i in range(ch_z)])
            twos_big = tf.pad(twos, [[0, 0], [1, 0], [0, 1]], mode='CONSTANT')

            L = ones + twos_big
            L_T = tf.transpose(L, perm=[0, 2, 1])
            prec = tf.matmul(L, L_T)

            cov = tf.linalg.inv(prec)

            self._prior = tfd.MultivariateNormalFullCovariance(loc=zeros, covariance_matrix=cov, name="prior")
        else:
            raise ValueError("specified string is not a suitable variational layer: %s" % str(self._variational_layer))

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
            # reduced = tf.layers.Conv1D(filters=self._latent_channels,
            #                            kernel_size=self._hop_length,
            #                            strides=self._hop_length,
            #                            padding='same')(inputs)

            reduced = snt.Conv1D(output_channels=self._latent_channels,
                                kernel_shape=self._hop_length,
                                stride=self._hop_length,
                                padding=snt.CAUSAL,
                                name='dim_reduction_conv')(inputs)

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

    def get_gaussian(self, inputs):
        if self._variational_layer == GAUSSIAN_DIAGONAL:
            var_lay = GaussianDiagonal(module_tuple=("Conv1D", {
                "stride":       1,
                "kernel_shape": self._filter_length,}),
                                       # {"pool_size": self._hop_length}
                                       output_shape=self.en.shape,
                                       minimal_covariance=0.,
                                       covariance_parameterization="softplus",
                                       scalar_covariance=False, )
        elif self._variational_layer == GAUSSIAN_TRI_DIAGONAL:
            var_lay = GaussianTriDiagonal(module_tuple=("Conv1D", {
                "stride":       1,
                "kernel_shape": self._filter_length,}),
                                          # {"pool_size": self._hop_length}
                                          output_shape=self.en.shape,
                                          minimal_covariance=0.,
                                          covariance_parameterization="softplus",
                                          scalar_covariance=False, )
        elif self._variational_layer == GAUSSIAN_TRI_DIAGONAL_PRECISION:
            var_lay = GaussianTriDiagonalPrecision(module_tuple=("Conv1D", {
                "stride":       1,
                "kernel_shape": self._filter_length,}),
                                                   # {"pool_size": self._hop_length}
                                                   output_shape=self.en.shape,
                                                   minimal_covariance=0.1,
                                                   covariance_parameterization="softplus",
                                                   scalar_covariance=False, )
        else:
            raise ValueError("specified string is not a suitable variational layer: %s" % str(self._variational_layer))

        return var_lay(inputs)

