import numpy as np
import tensorflow as tf

from core.optimizers.linalg.woodberry import _woodberry, _woodberry_alpha_trick

sess = tf.Session()


def test_woo1():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[1.], [2.], [3.]]), dtype=tf.float64)
    C = tf.constant(np.asarray([1.]), dtype=tf.float64)
    V = tf.transpose(U, perm=[1, 0])
    I = tf.eye(3, dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, tf.matmul(tf.linalg.diag(C), V))
    compare = tf.linalg.inv(v_)

    inverse = _woodberry(alpha=alpha, U=U, C=C, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_woo2():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[1., 2.], [3., 4.], [5., 6.]]), dtype=tf.float64)
    C = tf.constant(np.asarray([1., 1.]), dtype=tf.float64)
    V = tf.transpose(U, perm=[1, 0])
    I = tf.eye(3, dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, tf.matmul(tf.linalg.diag(C), V))
    compare = tf.linalg.inv(v_)

    inverse = _woodberry(alpha=alpha, U=U, C=C, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_woo3():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[1., 2.], [3., 4.], [5., 6.]]), dtype=tf.float64)
    C = tf.constant(np.asarray([2., 2.]), dtype=tf.float64)
    V = tf.transpose(U, perm=[1, 0])
    I = tf.eye(3, dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, tf.matmul(tf.linalg.diag(C), V))
    compare = tf.linalg.inv(v_)

    inverse = _woodberry(alpha=alpha, U=U, C=C, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_woo4():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[1., 2.], [3., 4.], [5., 6.]]), dtype=tf.float64)
    C = tf.constant(np.asarray([[2., 2.]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[1, 0])
    I = tf.eye(3, dtype=tf.float64)

    G = tf.ones([1, 3], dtype=tf.float64)

    v_ = (alpha * I + tf.einsum("ik, lkj-> lij", U, tf.einsum("lik, kj-> lij", tf.linalg.diag(C), V))) / (1. + alpha)

    compare = tf.einsum("lik, lk-> li", tf.linalg.inv(v_), G)

    inverse = _woodberry_alpha_trick(U=U, C=C, V_T=V, alpha=alpha, grads=G)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)
