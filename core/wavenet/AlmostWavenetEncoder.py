import tensorflow as tf
import sonnet as snt

from .utils import pool1d, mu_law

class AlmostWavenetEncoder(snt.AbstractModule):
    """AlmostWavenetEncoder
    .........
    """

    def __init__(self, num_layers_per_stage,
                 num_layers,
                 filter_length,
                 hop_length,
                 e_hidden_channels,
                 latent_channels,
                 name='almost_wavenet_encoder'):
        super(AlmostWavenetEncoder, self).__init__(name=name)
        self._num_layers_per_stage = num_layers_per_stage
        self._num_layers = num_layers
        self._filter_length = filter_length
        self._hop_length = hop_length
        self._e_hidden_channels = e_hidden_channels
        self._latent_channels = latent_channels

    def _build(self, inputs):
        """Build the graph for this configuration.

       Args:
         inputs: input node (already preprocessed)

       Returns:
         A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
         the 'quantized_input', and whatever metrics we want to track for eval.
       """

        # https://deepmind.github.io/sonnet/_modules/sonnet/python/modules/conv.html#Conv1D

        ###
        # The Non-Causal Temporal Encoder.
        ###
        ae_startconv = snt.Conv1D(output_channels=self._e_hidden_channels,
                                  kernel_shape=self._filter_length,
                                  name='ae_startconv')

        en = ae_startconv(inputs)

        for num_layer in range(self._num_layers):
            dilation = 2 ** (num_layer % self._num_layers_per_stage)
            d = tf.nn.relu(en)

            ae_dilatedconv = snt.Conv1D(output_channels=self._e_hidden_channels,
                                        kernel_shape=self._filter_length,
                                        rate=dilation,
                                        name='ae_dilatedconv_%d' % (num_layer + 1))

            d = ae_dilatedconv(d)

            d = tf.nn.relu(d)

            ae_res = snt.Conv1D(output_channels=self._e_hidden_channels,
                                kernel_shape=1,
                                padding=snt.CAUSAL,
                                name='ae_res_%d' % (num_layer + 1))

            en += ae_res(d)

        ae_bottleneck = snt.Conv1D(output_channels=self._latent_channels,
                                   kernel_shape=1,
                                   padding=snt.CAUSAL,
                                   name='ae_bottleneck')

        en = ae_bottleneck(en)

        return en
