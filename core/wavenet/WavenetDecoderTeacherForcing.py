import sonnet as snt
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .utils import condition, inv_mu_law

class WavenetDecoderTeacherForcing(snt.AbstractModule):
    """The learned decoder distribution p(x|z).

      The output distribution will be a Bernoulli distribution for each pixel,
      with the mean of the distribution being learned by gradient descent.

      The variables of this module get a loss from the likelihood term and are
      trained using gradient descent.
    """

    def __init__(self, num_layers_per_stage,
                 num_layers,
                 filter_length,
                 d_hidden_channels,
                 skip_channels,
                 prob_dropout_decoder_tf,
                 name='wavenet_decoder_teacher_forcing'):
        super(WavenetDecoderTeacherForcing, self).__init__(name=name)
        self._num_layers_per_stage = num_layers_per_stage
        self._num_layers = num_layers
        self._filter_length = filter_length
        self._d_hidden_channels = d_hidden_channels
        self._skip_channels = skip_channels
        self._prob_dropout_decoder_tf = prob_dropout_decoder_tf

    def _build(self, x_shifted, en, conditioning=None):

        self._nodes_list = []

        startconv = snt.Conv1D(output_channels=self._d_hidden_channels,
                               kernel_shape=self._filter_length,
                               padding=snt.CAUSAL,
                               name='startconv')

        skip_start = snt.Conv1D(output_channels=self._skip_channels,
                                kernel_shape=1,
                                padding=snt.CAUSAL,
                                name='skip_start')

        if self._prob_dropout_decoder_tf > 0:
            l = tf.nn.dropout(x_shifted, rate=self._prob_dropout_decoder_tf)
            l = startconv(x_shifted)
        else:
            l = startconv(x_shifted)

        self._nodes_list.append(l)

        # Set up skip connections.
        s = skip_start(l)
        self._nodes_list.append(s)

        # Residual blocks with skip connections.
        for i in range(self._num_layers):
            dilation = 2 ** (i % self._num_layers_per_stage)

            causal_convolution = snt.Conv1D(output_channels=2 * self._d_hidden_channels,
                                            kernel_shape=[self._filter_length],
                                            rate=dilation,
                                            padding=snt.CAUSAL,
                                            name='dilatedconv_%d' % (i + 1))

            dil = causal_convolution(l)
            self._nodes_list.append(dil)

            encoding_convolution = snt.Conv1D(output_channels=2 * self._d_hidden_channels,
                                              kernel_shape=1,
                                              padding=snt.CAUSAL,
                                              name='cond_map_%d' % (i + 1))

            enc = encoding_convolution(en)
            dil = condition(dil, enc)
            self._nodes_list.append(dil)

            assert dil.get_shape().as_list()[2] % 2 == 0
            m = dil.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(dil[:, :, :m])
            d_tanh = tf.tanh(dil[:, :, m:])
            dil = d_sigmoid * d_tanh

            res_convolution = snt.Conv1D(output_channels=self._d_hidden_channels,
                                         kernel_shape=1,
                                         padding=snt.CAUSAL,
                                         name='res_%d' % (i + 1))

            l += res_convolution(dil)
            self._nodes_list.append(l)

            skip_conv = snt.Conv1D(output_channels=self._skip_channels,
                                   kernel_shape=1,
                                   padding=snt.CAUSAL,
                                   name='skip_%d' % (i + 1))

            s += skip_conv(dil)
            self._nodes_list.append(s)


        s = tf.nn.relu(s)

        out = snt.Conv1D(output_channels=self._skip_channels,
                         kernel_shape=1,
                         padding=snt.CAUSAL,
                         name='out1')

        s = out(s)
        self._nodes_list.append(s)

        cond_map_out = snt.Conv1D(output_channels=self._skip_channels,
                                  kernel_shape=1,
                                  padding=snt.CAUSAL,
                                  name='cond_map_out1')


        s = condition(s, cond_map_out(en))

        s = tf.nn.relu(s)
        self._nodes_list.append(s)

        logits_conv = snt.Conv1D(output_channels=256,
                                 kernel_shape=1,
                                 padding=snt.CAUSAL,
                                 name='logits')

        self.logits = logits_conv(s)

        # self.logits = tf.reshape(self.logits, [-1, 256])
        # self._probs = tf.nn.softmax(self.logits, name='softmax')
        self.dec_distr = tfd.Categorical(logits=self.logits)
        # dimension of rec_sample should be: [bs, length, 1]

        self.reconstruction = inv_mu_law(tf.expand_dims(self.dec_distr.mode(), axis=-1) - 128)
        # self.reconstruction = inv_mu_law(tf.expand_dims(self.dec_distr.mean(), axis=-1) - 128)

        return self.dec_distr, self.reconstruction
