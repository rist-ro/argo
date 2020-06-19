from argo.core.flows.build_flow import build_flow
import tensorflow as tf
from tensorflow_probability import distributions as tfd

GAUSSIAN_TRI_DIAGONAL_PRECISION = "GaussianTriDiagonalPrecision"

GAUSSIAN_TRI_DIAGONAL = "GaussianTriDiagonal"

GAUSSIAN_DIAGONAL = "GaussianDiagonal"


class IndependentChannelsTransformedDistribution:
    def __init__(self, event_shape, flow_params, dim, channels):
        self._flow_params = flow_params
        self._dim = dim
        self._channels = channels
        base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(event_shape))

        self._independent_channels_distributions = []
        for ch in range(self._channels):
            flow_maf = build_flow(flow_params, flow_size=dim)
            transformed_distribution = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_maf)
            self._independent_channels_distributions.append(transformed_distribution)
        print()

    def mean(self):
        '''
        Returns: tensor of shape (ch, dim)
        '''
        samples_independent_channels = tf.concat([self._independent_channels_distributions[i].sample(100) for i in
                                                  range(self._channels)], axis=1)
        return tf.reduce_mean(samples_independent_channels, axis=0)

    def covariance(self):
        '''
        Returns: tensor of shape (ch, dim, dim)
        '''
        return tf.zeros([self._channels, self._dim, self._dim])

    def log_prob(self, data):
        '''

        Args:
            data (tf.Tensor): of shape (ch, dim) or (batch, ch, dim)
        Returns:
            log_prob: of shape (ch) if no batch shape given or (batch, ch)
        '''
        data_channels = data.get_shape().as_list()[-2]
        assert data_channels == self._channels,\
            'Channels in the data: {} and the distribution dont match on axis -2'.format(data_channels,
                                                                                         self._channels)
        return tf.concat(
            [self._independent_channels_distributions[i].log_prob(data[..., i:i + 1, :]) for i in
             range(self._channels)],
            axis=-1)

    def sample(self, n_samples):
        return tf.concat(
            [self._independent_channels_distributions[i].sample(n_samples) for i in range(self._channels)],
            axis=-2
        )
