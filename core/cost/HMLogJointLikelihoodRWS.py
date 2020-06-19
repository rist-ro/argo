import tensorflow as tf

from argo.core.network.AbstractModule import AbstractModule


class HMLogJointLikelihoodRWS(AbstractModule):

    def __init__(self, name="HMLJLRWS"):
        super().__init__(name=name)

    def create_id(self, cost_fuction_kwargs):

        _id = "HMLJLRWS"

        return _id

    def _build(self, hm):
        batch_size = hm.b_size
        n_samples = hm.n_z_samples
        mega_batch_size = batch_size * n_samples

        log_joint_likelihood_p = self._get_reconstruction_log_joint_likelihood_p(hm, mega_batch_size)

        log_joint_likelihood_q = self._get_dream_rec_log_joint_conditional_q(hm, mega_batch_size)

        log_px_unnormalized = hm.importance_sampling_node - tf.log(tf.cast(n_samples, dtype=tf.float32))

        total_loss = -tf.reduce_mean(log_px_unnormalized, axis=0)

        reconstruction_loss = -log_joint_likelihood_p
        dream_reconstruction_loss = -log_joint_likelihood_q

        return total_loss, [[reconstruction_loss], [dream_reconstruction_loss]], [["NLL_X"], ["NLL_H"]], [{
            "fileName": "reconstruction_loss_NLLX"}, {
            "fileName": "reconstruction_loss_NLLH"}]

    def _get_dream_rec_log_joint_conditional_q(self, hm, mega_batch_size):
        gen_layers = hm._hgs
        rec_layers = hm._hrs

        log_joint_likelihood_q = tf.zeros((mega_batch_size))

        for i in range(1, len(rec_layers)):
            sample_p = gen_layers[i][1]
            log_likelihood = rec_layers[i][0].log_prob(sample_p)

            log_joint_likelihood_q += tf.reduce_sum(log_likelihood, axis=-1)

        log_joint_likelihood_q = tf.reduce_mean(log_joint_likelihood_q, axis=0)

        return log_joint_likelihood_q

    def _get_reconstruction_log_joint_likelihood_p(self, hm, mega_batch_size):
        gen_layers = hm._hgw
        rec_layers = hm._hrw

        log_joint_likelihood_p = tf.zeros((mega_batch_size))

        for i in range(len(gen_layers)):
            sample_q = rec_layers[i][1]
            distr_p = gen_layers[i][0]

            log_joint_likelihood_p += tf.reduce_sum(distr_p.log_prob(sample_q), axis=-1)

        log_joint_likelihood_p = tf.reduce_mean(log_joint_likelihood_p, axis=0)

        return log_joint_likelihood_p
