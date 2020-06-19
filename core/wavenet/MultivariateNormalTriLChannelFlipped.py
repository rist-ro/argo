import tensorflow as tf
from tensorflow_probability import distributions as tfd

class MultivariateNormalTriLChannelFlipped(tfd.MultivariateNormalTriL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, sample_shape=(), seed=None, name="sample"):
        return tf.transpose(super().sample(sample_shape, seed, name),
                                           perm=[0, 1, 3, 2]) #extra dimension because of sample