import tensorflow as tf
import numpy as np


def mu_law(x, mu=255, int8=False):
    """A TF implementation of Mu-Law encoding.

    Args:
      x: The audio samples to encode between [-1, 1]
      mu: The Mu to use in our Mu-Law.
      int8: Use int8 encoding.

    Returns:
      out: The Mu-Law encoded int8 data [-128, 127].
    """
    out = tf.clip_by_value(x, -1, 0.999)
    out = tf.sign(out) * tf.log(1 + mu * tf.abs(out)) / np.log(1 + mu)
    out = tf.floor(out * 128)
    if int8:
        out = tf.cast(out, tf.int8)
    return out

def mu_law_numpy(x, mu=255, int8=False):
    """A TF implementation of Mu-Law encoding.

    Args:
      x: The audio samples to encode between [-1, 1]
      mu: The Mu to use in our Mu-Law.
      int8: Use int8 encoding.

    Returns:
      out: The Mu-Law encoded int8 data [-128, 127].
    """
    out = np.clip(x, -1, 0.999)
    out = np.sign(out) * np.log(1 + mu * np.abs(out)) / np.log(1 + mu)
    out = np.floor(out * 128)
    if int8:
        out = tf.cast(out, tf.int8)
    return out

def inv_mu_law(x, mu=255, name=None):
  """A TF implementation of inverse Mu-Law.

  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.

  Returns:
    out: The decoded data.
  """

  # this method expects input x as an int between [-128, 127]
  x = tf.cast(x, tf.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  # TODO I think it should be the following, to have out \in [-1,1]
  # out = (x + 0.5) * 2. / mu
  out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
  out = tf.where(tf.equal(x, 0), x, out, name=name)
  return out

def condition(x, encoding):
    """Condition the input on the encoding.

    Args:
      x: The [mb, length, channels] float tensor input.
      encoding: The [mb, encoding_length, channels] float tensor encoding.

    Returns:
      The output after broadcasting the encoding to x's shape and adding them.
    """

    mb = tf.shape(x)[0]
    length = tf.shape(x)[1]
    channels = x.get_shape().as_list()[2]

    enc_mb = tf.shape(encoding)[0]
    enc_length = tf.shape(encoding)[1]
    enc_channels = encoding.get_shape().as_list()[2]

    assert enc_channels == channels

    with tf.control_dependencies([tf.assert_equal(enc_mb, mb)]):
        encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
        x = tf.reshape(x, [mb, enc_length, -1, channels])
        x += encoding
        x = tf.reshape(x, [mb, length, channels])

    return x


def shift_right(x):
    """Shift the input over by one and a zero to the front.

    Args:
      x: The [mb, time, channels] tensor input.

    Returns:
      x_sliced: The [mb, time, channels] tensor output.
    """
    ch = x.get_shape().as_list()[2]
    length = tf.shape(x)[1]
    x_padded = tf.pad(x, [[0, 0], [1, 0], [0, 0]])
    x_sliced = tf.slice(x_padded, [0, 0, 0], tf.stack([-1, length, ch]))
    # x_sliced.set_shape(shape)
    return x_sliced


def pool1d(x, pool_size, name, mode='avg', stride=None):
    """1D pooling function that supports multiple different modes.

    Args:
      x: The [mb, time, channels] float tensor that we are going to pool over.
      window_length: The amount of samples we pool over.
      name: The name of the scope for the variables.
      mode: The type of pooling, either avg or max.
      stride: The stride length.

    Returns:
      pooled: The [mb, time // stride, channels] float tensor result of pooling.
    """

    if mode == 'avg':
        pool_fn = tf.layers.average_pooling1d
    elif mode == 'max':
        pool_fn = tf.layers.max_pooling1d
    else:
        raise TypeError("No such pooling function")


    stride = stride or pool_size
    # batch_size, length, num_channels = x.get_shape().as_list()
    length = tf.shape(x)[1]
    # assert length % window_length == 0
    # assert length % stride == 0

    with tf.control_dependencies([tf.assert_equal(tf.mod(length, pool_size), 0),
                                  tf.assert_equal(tf.mod(length, stride), 0)
                                  ]):
        pooled = pool_fn(x, pool_size, stride, padding='VALID', name=name)
        return pooled


def empty_all_queues(sess, queues_size, queues_dequeue):
    # smart dequeuing contemporaneously on all queues till I can
    all_sizes = np.array(sess.run(queues_size))
    while np.sum(all_sizes)>0:
        idxs = np.where(all_sizes > 0)[0]
        dequeue_ops = list(np.array(queues_dequeue)[idxs])
        sess.run(dequeue_ops)
        all_sizes = np.array(sess.run(queues_size))

    return all_sizes