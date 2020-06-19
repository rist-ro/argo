import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from core.optimizers.linalg.shermann_morrison import _shermann_morison_alpha_trick_single, _shermann_morison_alpha_trick
from core.optimizers.linalg.woodberry import _woodberry_alpha_trick

n = 200
k = 50
l = 2


def test_speed_SM_S():
    alpha = tf.constant(0.1, dtype=tf.float64)

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.diag(np.random.rand(k)), dtype=tf.float64)
    I = tf.eye(n, dtype=tf.float64)

    start_setup = time.time()

    U_hat = tf.matmul(U, tf.sqrt(Q))
    V = tf.transpose(U_hat, perm=[1, 0])

    inverse = _shermann_morison_alpha_trick_single(alpha=alpha, U=U_hat)

    end_setup = time.time()

    v_ = (alpha * I + tf.matmul(U_hat, V)) / (1. + alpha)
    compare = tf.linalg.inv(v_)

    sess = tf.Session()

    start_run = time.time()
    c, ii = 0, 0

    for i in range(100):
        c, ii = sess.run([compare, inverse])

    end_run = time.time()
    print("Setup time SM-s", timedelta(seconds=end_setup - start_setup))
    print("Run time SM-s", timedelta(seconds=end_run - start_run))

    assert np.allclose(c, ii), "Not close"


# This version is consistently the fastest
def test_speed_SM():
    alpha = tf.constant(0.1, dtype=tf.float64)

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(l, k), dtype=tf.float64)
    I = tf.eye(n, dtype=tf.float64)

    start_setup = time.time()

    V_T = tf.transpose(U, perm=[1, 0])
    V_hat = tf.einsum("lik, kj->lij", tf.linalg.diag(Q), V_T)

    G = tf.ones([l, n], dtype=tf.float64)
    inverse = _shermann_morison_alpha_trick(alpha=alpha, U=U, V_T=V_hat, grads=G)

    end_setup = time.time()

    v_ = (alpha * I + tf.einsum("ik, lkj->lij", U, V_hat)) / (1. + alpha)
    compare = tf.einsum("lik, lk->li", tf.linalg.inv(v_), G)

    sess = tf.Session()
    c, ii = 0, 0
    start_run = time.time()
    for i in range(100):
        c, ii = sess.run([compare, inverse])

    end_run = time.time()
    print("Setup time SM", timedelta(seconds=end_setup - start_setup))
    print("Run time SM", timedelta(seconds=end_run - start_run))

    assert np.allclose(c, ii), "Not close"


def test_speed_W():
    alpha = tf.constant(0.1, dtype=tf.float64)

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(l, k), dtype=tf.float64)
    I = tf.eye(n, dtype=tf.float64)

    start_setup = time.time()

    V_T = tf.transpose(U, perm=[1, 0])
    V_hat = tf.einsum("lik, kj->lij", tf.linalg.diag(Q), V_T)

    G = tf.ones([l, n], dtype=tf.float64)
    inverse = _woodberry_alpha_trick(U=U, C=Q, V_T=V_T, alpha=alpha, grads=G)

    end_setup = time.time()

    v_ = (alpha * I + tf.einsum("ik, lkj->lij", U, V_hat)) / (1. + alpha)
    compare = tf.einsum("lik, lk->li", tf.linalg.inv(v_), G)

    sess = tf.Session()

    start_run = time.time()
    c, ii = 0, 0

    for i in range(100):
        c, ii = sess.run([compare, inverse])

    end_run = time.time()
    print("Setup time Woodberry", timedelta(seconds=end_setup - start_setup))
    print("Run time Woodberry", timedelta(seconds=end_run - start_run))

    assert np.allclose(c, ii), "Not close"
