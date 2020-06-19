import tensorflow as tf
from tensorflow.python.ops import array_ops

from argo.core.utils.argo_utils import NUMTOL

from argo.core.Network import AbstractModule

class SR(AbstractModule):

    def __init__(self, samples=1000, baseline=0, name="SR"):
        super().__init__(name = name)

        self._samples = samples
        self._baseline = baseline == 1
        
    @staticmethod
    def create_id(cost_fuction_kwargs):
        return "SR_s" + str(cost_fuction_kwargs["samples"]) + "_" + str(cost_fuction_kwargs["baseline"])

    def _build(self, ae):

        model_visible = ae._model_visible
        x_target = ae.x_target
        reconstruction = ae.x_reconstruction_node

        # this has changed since some models can feed data with different shapes for train and eval
        # x_shape is also a dict alternatively... if this break ask me.. (Riccardo)
        # dims = np.prod(ae.x_shape)
        dims = tf.cast(tf.reduce_prod(tf.shape(self.raw_x)[1:]),
                                    tf.float32)
        # dims = np.prod(ae.x_shape["train"])

        # sample
        x_prime = tf.cast(model_visible.sample(sample_shape=[self._samples]), dtype=tf.float32)
        x_target = tf.cast(x_target, dtype=tf.float32)

        #x_prime = tf.check_numerics(x_prime, "x_prime is not finite")
        #x_target = tf.check_numerics(x_target, "x_target is not finite")

        x_prime = tf.clip_by_value(x_prime, NUMTOL, 1-NUMTOL)
        
        norm = tf.norm(tf.reshape(x_target - x_prime, [self._samples, -1, dims]), axis=2)

        all_log_probs = model_visible.log_prob(x_prime)
        #all_log_probs = tf.check_numerics(all_log_probs, "all_log_probs is not finite")
        
        log_likelihood = tf.reduce_sum(tf.reshape(all_log_probs, [self._samples, -1, ae.x_shape[0], ae.x_shape[1]]), axis=[2, 3])
        #log_likelihood = tf.check_numerics(log_likelihood, "log_likelihood is not finite")
        
        if self._baseline:
            loss_factor = norm - tf.reduce_mean(norm, axis=0)  # subtract baseline
        else:
            loss_factor = norm
                    
        loss_factor = array_ops.stop_gradient(loss_factor)
        loss_batch = tf.multiply(log_likelihood, loss_factor)

        mean_per_reconstructions = tf.reduce_mean(loss_batch, axis=1)
        loss = tf.reduce_mean(mean_per_reconstructions)
        
        # compute true cost function (L2)
        norms = tf.norm(tf.reshape(x_target - reconstruction,[-1, dims]),axis=1)
        l2 = 1/2 * tf.reduce_mean(tf.square(norms))

        return loss, [l2], ["L2"]

    
