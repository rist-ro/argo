from collections import namedtuple

import numpy as np
import tensorflow as tf

from core.optimizers.NaturalWakeSleepOptimizer import NaturalWakeSleepOptimizer
from core.optimizers.NaturalWakeSleepOptimizerAlternate import NaturalWakeSleepOptimizerAlternate
from core.optimizers.linalg.woodberry import _optimized_woodberry, _woodberry_alpha_trick


def get_fake_model(n=100, b=4, s=5):
    fake_model = {
        "b_size":      b,
        "batch_size":  {
            "train": b},
        "samples":     s,
        "n_z_samples": s}
    fake_model = namedtuple('FHM', fake_model.keys())(*fake_model.values())

    k = fake_model.b_size * fake_model.samples
    optimizer_kwargs = {
        "model":                    fake_model,
        "individual_learning_rate": 1.0,
        "learning_rate":            0.01,
        "rescale_learning_rate":    1.0,
        "diagonal_pad":             0.001,
        'k_step_update':            1}

    global_step = tf.Variable(1)
    return k, fake_model, optimizer_kwargs, global_step, n, b, s


def test_speed_N_vs_NA():
    k, fake_model, optimizer_kwargs, global_step, n, b, s = get_fake_model(n=100)

    G = tf.constant(np.random.rand(n), dtype=tf.float64, shape=[1, n])

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(k), dtype=tf.float64, shape=[1, k])

    NWSO = NaturalWakeSleepOptimizer(**optimizer_kwargs)
    NWSO._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSO._diagonal_cond = tf.less_equal(NWSO._diagonal_pad, 10.0)

    NWSOAL = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL._diagonal_cond = tf.less_equal(NWSOAL._diagonal_pad, 10.0)

    inverse = NWSO._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer1")
    compare = NWSOAL._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer2")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c, ii = sess.run([compare, inverse])
    print(c[:10, :10])
    print(ii[:10, :10])
    assert np.allclose(c, ii, ), "Not that close"


def test_speed_N_vs_NA_small():
    k, fake_model, optimizer_kwargs, global_step, n, b, s = get_fake_model(n=10)

    G = tf.constant(np.random.rand(n), dtype=tf.float64, shape=[1, n])

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(k), dtype=tf.float64, shape=[1, k])

    NWSO = NaturalWakeSleepOptimizer(**optimizer_kwargs)
    NWSO._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSO._diagonal_cond = tf.less_equal(NWSO._diagonal_pad, 10.0)

    NWSOAL = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL._diagonal_cond = tf.less_equal(NWSOAL._diagonal_pad, 10.0)

    inverse = NWSO._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer1")
    compare = NWSOAL._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer2")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c, ii = sess.run([compare, inverse])
    print(c[:10, :10])
    print(ii[:10, :10])
    assert np.allclose(c, ii, ), "Not that close"


def test_speed_N_vs_NA_big():
    k, fake_model, optimizer_kwargs, global_step, n, b, s = get_fake_model(n=500)

    G = tf.constant(np.random.rand(n), dtype=tf.float64, shape=[1, n])

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(k), dtype=tf.float64, shape=[1, k])

    NWSO = NaturalWakeSleepOptimizer(**optimizer_kwargs)
    NWSO._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSO._diagonal_cond = tf.less_equal(NWSO._diagonal_pad, 10.0)

    NWSOAL = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL._diagonal_cond = tf.less_equal(NWSOAL._diagonal_pad, 10.0)

    inverse = NWSO._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer1")
    compare = NWSOAL._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer2")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c, ii = sess.run([compare, inverse])
    print(c[:10, :10])
    print(ii[:10, :10])
    assert np.allclose(c, ii, ), "Not that close"


def test_speed_N_vs_NA_Wclean():
    k, _, _, _, n, _, _ = get_fake_model(n=500)

    alpha = tf.constant(0.1, dtype=tf.float64)

    G = tf.constant(np.random.rand(n), dtype=tf.float64, shape=[1, n])

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(k), dtype=tf.float64, shape=[1, k])

    U_T = tf.transpose(U)

    backward = _optimized_woodberry(U=U, C=Q, V_T=U_T, alpha=alpha, grads=G)
    forward = _woodberry_alpha_trick(U=U, C=Q, V_T=U_T, alpha=alpha, grads=G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c, ii = sess.run([forward, backward])
    print(c[:10, :10])
    print(ii[:10, :10])
    assert np.allclose(c, ii, ), "Not that close"

def test_speed_NA_SH_vs_INV_W_big():
    k, fake_model, optimizer_kwargs, global_step, n, b, s = get_fake_model(n=500)

    G = tf.constant(np.random.rand(n), dtype=tf.float64, shape=[1, n])

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(k), dtype=tf.float64, shape=[1, k])

    NWSOAL1 = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL1._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL1._diagonal_cond = tf.less_equal(NWSOAL1._diagonal_pad, 10.0)

    NWSOAL2 = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL2._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL2._diagonal_cond = tf.less_equal(NWSOAL2._diagonal_pad, 10.0)

    inverse = NWSOAL1._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer1", choice=True)
    compare = NWSOAL2._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer2", choice=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c, ii = sess.run([compare, inverse])
    print(c[:10, :10])
    print(ii[:10, :10])
    assert np.allclose(c, ii, ), "Not that close"

def test_speed_NA_SH_vs_INV_W_small():
    k, fake_model, optimizer_kwargs, global_step, n, b, s = get_fake_model(n=10)

    G = tf.constant(np.random.rand(n), dtype=tf.float64, shape=[1, n])

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(k), dtype=tf.float64, shape=[1, k])

    NWSOAL1 = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL1._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL1._diagonal_cond = tf.less_equal(NWSOAL1._diagonal_pad, 10.0)

    NWSOAL2 = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL2._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL2._diagonal_cond = tf.less_equal(NWSOAL2._diagonal_pad, 10.0)

    inverse = NWSOAL1._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer1", choice=True)
    compare = NWSOAL2._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer2", choice=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c, ii = sess.run([compare, inverse])
    print(c[:10, :10])
    print(ii[:10, :10])
    assert np.allclose(c, ii, ), "Not that close"

def test_speed_NA_SH_vs_INV_W_RIcc():
    k, fake_model, optimizer_kwargs, global_step, n, b, s = get_fake_model(n=500, b=10, s=1)

    G = tf.constant(np.random.rand(n), dtype=tf.float64, shape=[1, n])

    U = tf.constant(np.random.rand(n, k), dtype=tf.float64)
    Q = tf.constant(np.random.rand(k), dtype=tf.float64, shape=[1, k])

    NWSOAL1 = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL1._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL1._diagonal_cond = tf.less_equal(NWSOAL1._diagonal_pad, 10.0)

    NWSOAL2 = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL2._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL2._diagonal_cond = tf.less_equal(NWSOAL2._diagonal_pad, 10.0)

    inverse = NWSOAL1._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer1", choice=True)
    compare = NWSOAL2._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer2", choice=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c, ii = sess.run([compare, inverse])
    print(c[:10, :10])
    print(ii[:10, :10])
    assert np.allclose(c, ii, ), "Not that close"

def test_speed_NA_SH_vs_INV_W_Ricc2():
    k, fake_model, optimizer_kwargs, global_step, n, b, s = get_fake_model(n=500, b=10, s=1)

    G = tf.constant(np.random.normal(1,0.01, n), dtype=tf.float64, shape=[1, n])

    U = tf.constant(np.random.normal(1,0.01,n * k).reshape([n,k]), dtype=tf.float64)
    Q = tf.constant(np.random.normal(1,0.01, k), dtype=tf.float64, shape=[1, k])

    NWSOAL1 = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL1._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL1._diagonal_cond = tf.less_equal(NWSOAL1._diagonal_pad, 10.0)

    NWSOAL2 = NaturalWakeSleepOptimizerAlternate(**optimizer_kwargs)
    NWSOAL2._diagonal_pad = optimizer_kwargs["diagonal_pad"]
    NWSOAL2._diagonal_cond = tf.less_equal(NWSOAL2._diagonal_pad, 10.0)

    inverse = NWSOAL1._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer1", choice=True)
    compare = NWSOAL2._multiply_grads_by_fisher_inv(G, Q, U, global_step, "layer2", choice=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    c, ii = sess.run([compare, inverse])
    print(c[:10, :10])
    print(ii[:10, :10])
    assert np.allclose(c, ii, ), "Not that close"