import tensorflow as tf


def _damped_fisher_inverse(U, Q, U_T, alpha):

    V_T = tf.einsum('lk,kn->lkn', Q, U_T)
    F = tf.einsum('ij,ljk->lik', U, V_T)

    F_hat = alpha * tf.eye(F.shape.as_list()[-1], dtype=F.dtype) + F

    F_inv = tf.linalg.inv(F_hat)
    return (1.0 + alpha) * F_inv


def _true_fisher_inverse(U, Q, U_T):

    V_T = tf.einsum('lk,kn->lkn', Q, U_T)
    F = tf.einsum('ij,ljk->lik', U, V_T)

    F_inv = tf.linalg.inv(F)
    return F_inv
