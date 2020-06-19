import tensorflow as tf


def _shermann_morison_alpha_trick_single(alpha, U):
    V_T = tf.transpose(U, perm=[1, 0])
    n_z_s = tf.shape(U)[-1]
    k_layer_length = U.shape.as_list()[-2]

    M = tf.eye(k_layer_length, dtype=alpha.dtype)

    M_inner = alpha * tf.eye(n_z_s, dtype=alpha.dtype) + tf.matmul(V_T, U)
    M2 = tf.matmul(tf.matmul(U, tf.linalg.inv(M_inner)), V_T)

    M -= M2
    return ((1.0 + alpha) / alpha) * M


def _shermann_morison_alpha_trick(alpha, U, V_T, grads):
    n_z_s = tf.shape(U)[-1]

    M_inner_inv = tf.linalg.inv(alpha * tf.eye(n_z_s, dtype=alpha.dtype) + tf.einsum('lij,jk->lik', V_T, U))

    M2 = tf.einsum('ij,lj->li', U,
                   tf.einsum('lij,lj->li', M_inner_inv, tf.einsum('lik,lk->li', V_T, grads)))

    inverse_x_dif = grads - M2

    return ((1.0 + alpha) / alpha) * inverse_x_dif


def _shermann_morison(alpha, U, V_T):
    b_size = tf.shape(U)[0]
    n_z_s = tf.shape(U)[-1]
    k_layer_length = U.shape.as_list()[-2]

    M = 1.0 / alpha * tf.eye(k_layer_length, batch_shape=[b_size], dtype=alpha.dtype)

    M_inner = tf.eye(n_z_s, batch_shape=[b_size], dtype=alpha.dtype) + 1.0 / alpha * tf.matmul(V_T, U)
    M2 = tf.matmul(tf.matmul(U, tf.linalg.inv(M_inner)), V_T)

    M -= (1.0 / (alpha ** 2)) * M2

    return M


def _shermann_morison_nobatch(alpha, U, V_T):
    n_z_s = tf.shape(U)[-1]
    k_layer_length = U.shape.as_list()[-2]

    M = 1.0 / alpha * tf.eye(k_layer_length, dtype=alpha.dtype)

    M_inner = tf.eye(n_z_s, dtype=alpha.dtype) + 1.0 / alpha * tf.matmul(V_T, U)
    M2 = tf.matmul(tf.matmul(U, tf.linalg.inv(M_inner)), V_T)

    M -= (1.0 / (alpha ** 2)) * M2

    return M
