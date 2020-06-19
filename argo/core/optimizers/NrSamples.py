import tensorflow as tf

def process_n_samples(n_samples, global_epoch, name="n_samples"):
    """
    Args:
        global_epoch:
        n_samples: can be either a number, a dictionary {"step": n_s}, e.g. {100:0.1, 1000:0.001} or a particular keyword

    Returns:
        the processed learning rate, ready to be taken by the optimizer (if a schedule was requested returns a tf node)
    """

    n_s = None

    if isinstance(n_samples, (int)):
        n_s = tf.placeholder_with_default(n_samples, shape=(), name='n_samples')

    # instantiate n_s node if n_s is None and n_samples is a dict at this point
    if n_s is None and isinstance(n_samples, dict):
        if not 0 in n_samples:
            raise ValueError(
                "learning rate schedule must specify, learning rate for step 0. Found schedule: %s" % n_samples)

        n_s = tf.constant(n_samples[0], dtype=tf.int32, name=name)
        for key, value in n_samples.items():
            n_s = tf.cond(
                tf.less(global_epoch, key), lambda: n_s, lambda: tf.constant(value, dtype=tf.int32, name=name))
        tf.summary.scalar("n_samples", n_s)

    if n_s is None:
        raise Exception("oops, something went wrong... could not process learning rate {}".format(str(n_samples)))

    return tf.identity(n_s, name="n_samples")


def get_n_s_id(n_samples):
    """
    Args:
        n_samples: can be either a number, a dictionary {"step": n_s}, e.g. {100:0.1, 1000:0.001} or a particular keyword,
                    or a tuple (n_s_min, n_s_name, n_s_kwargs)

    Returns:
        the id for the learning rate
    """

    _id = ""
    if isinstance(n_samples, (int)):
        _id += str(n_samples)

    elif isinstance(n_samples, dict):
        keys=n_samples.keys()
        val=n_samples.values()
        name = []
        for k,v in zip(keys,val):
            name.append("{}r{}".format(k,v))
        n = "_".join(name)

        _id += n

    return _id

