import numpy as np
import tensorflow as tf

from core.optimizers.linalg.shermann_morrison import _shermann_morison, _shermann_morison_nobatch, \
    _shermann_morison_alpha_trick, _shermann_morison_alpha_trick_single

sess = tf.Session()


def test_sh1():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.eye(2, batch_shape=[1], dtype=tf.float64)
    V = tf.transpose(U, perm=[0, 2, 1])
    I = tf.eye(2, batch_shape=[1], dtype=tf.float64)

    matmul = tf.matmul(U, V)
    alpha_i = alpha * I
    v_ = alpha_i + matmul
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison(alpha=alpha, U=U, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh2():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[[1., 2.], [2, 1]]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[0, 2, 1])
    I = tf.eye(2, batch_shape=[1], dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, V)
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison(alpha=alpha, U=U, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh3():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[[1., 2.], [2, 1], [0, 1]]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[0, 2, 1])
    I = tf.eye(3, batch_shape=[1], dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, V)
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison(alpha=alpha, U=U, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh4():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[[1.], [2.], [3.]]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[0, 2, 1])
    I = tf.eye(3, batch_shape=[1], dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, V)
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison(alpha=alpha, U=U, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh5():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[[1.], [2.], [3.]]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[0, 2, 1])
    I = tf.eye(3, batch_shape=[1, 3, 3], dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, V)
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison(alpha=alpha, U=U, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh6():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[[1.], [2.], [3.]], [[1.], [2.], [3.]]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[0, 2, 1])
    I = tf.eye(3, batch_shape=[2], dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, V)
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison(alpha=alpha, U=U, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh7():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[1.], [2.], [3.]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[1, 0])
    I = tf.eye(3, dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, V)
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison_nobatch(alpha=alpha, U=U, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh8():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[1., 2.], [3., 4.], [5., 6.]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[1, 0])
    I = tf.eye(3, dtype=tf.float64)

    v_ = alpha * I + tf.matmul(U, V)
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison_nobatch(alpha=alpha, U=U, V_T=V)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh9():
    alpha = tf.constant(0.1, dtype=tf.float64)

    U = tf.constant(np.asarray([[1., 2.], [3., 4.], [5., 6.]]), dtype=tf.float64)
    V = tf.transpose(U, perm=[1, 0])
    Q = tf.constant(np.asarray([[[1., 0.], [0., 2.]]]), dtype=tf.float64)
    V_hat = tf.einsum("lik, kj-> lij", Q, V)
    I = tf.eye(3, dtype=tf.float64)

    G = tf.ones([1, 3], dtype=tf.float64)

    v_ = (alpha * I + tf.einsum("ik, lkj-> lij", U, V_hat)) / (1. + alpha)
    compare = tf.einsum("lik, lk-> li", tf.linalg.inv(v_), G)

    inverse = _shermann_morison_alpha_trick(alpha=alpha, U=U, V_T=V_hat, grads=G)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)


def test_sh10():
    alpha = tf.constant(0.1, dtype=tf.float64)
    U = tf.constant(np.asarray([[1., 2.], [3., 4.], [5., 6.]]), dtype=tf.float64)
    Q = tf.constant(np.asarray([[1., 0.], [0., 2.]]), dtype=tf.float64)
    I = tf.eye(3, dtype=tf.float64)
    U_hat = tf.matmul(U, tf.sqrt(Q))
    V = tf.transpose(U_hat, perm=[1, 0])

    v_ = (alpha * I + tf.matmul(U_hat, V)) / (1. + alpha)
    compare = tf.linalg.inv(v_)

    inverse = _shermann_morison_alpha_trick_single(alpha=alpha, U=U_hat)

    c, i = sess.run([compare, inverse])

    assert np.allclose(c, i)
