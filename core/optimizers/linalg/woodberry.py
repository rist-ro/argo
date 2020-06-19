import tensorflow as tf


def _woodberry(alpha, U, C, V_T):
    k_layer_length = U.shape.as_list()[-2]

    C_inv = tf.map_fn(lambda x: 1.0 / x, C)
    C_inv_diag = tf.linalg.diag(C_inv)

    M = 1.0 / alpha * tf.eye(k_layer_length, dtype=alpha.dtype)

    M_inner = C_inv_diag + 1.0 / alpha * tf.matmul(V_T, U)
    M2 = tf.matmul(tf.matmul(U, tf.linalg.inv(M_inner)), V_T)

    M -= (1.0 / (alpha ** 2)) * M2

    return M


def _woodberry_alpha_trick(U, C, V_T, alpha, grads, layer=""):
    k_layer_length = U.shape.as_list()[-2]

    C_inv = tf.map_fn(lambda x: 1.0 / x, C)
    C_inv_diag = tf.linalg.diag(C_inv)

    M = tf.eye(k_layer_length, dtype=alpha.dtype)

    M_inner = alpha * C_inv_diag + tf.matmul(V_T, U)
    M2 = tf.einsum("lik, kj -> lij", tf.einsum("ik, lkj -> lij", U, tf.linalg.inv(M_inner)), V_T)

    M -= M2
    inverse_F = ((1.0 + alpha) / alpha) * M

    inverse_x_diff = tf.einsum("lik,lk->li", inverse_F, grads)

    return inverse_x_diff


def _optimized_woodberry(U, C, V_T, alpha, grads, layer=""):
    C_inv = tf.linalg.diag(1.0 / C)

    M_inner_inv = tf.linalg.inv(alpha * C_inv + tf.einsum('ij,jk->ik', V_T, U))

    M2 = tf.einsum('ij,lj->li', U,
                   tf.einsum('lij,lj->li', M_inner_inv, tf.einsum('ik,lk->li', V_T, grads)))

    M = grads - M2

    inverse_x_dif = ((1.0 + alpha) / alpha) * M

    return inverse_x_dif
