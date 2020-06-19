import numbers
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

def sum_regularizer(scale, scope = None):
    """Returns a function that can be used to apply sum regularization to weights.
    N.B. Sum regularization is very unortodox, it encourages highly negative weights.

    Args:
      scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
      scope: An optional scope name.
    Returns:
      A function with signature `sum_reg(weights)` that apply sum regularization.
    Raises:
      ValueError: If scale is not a float.
    """

    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)
    if isinstance(scale, numbers.Real):
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _: None

    def sum_reg(weights, name=None):
        """Applies sum regularization to weights."""
        with ops.name_scope(scope, 'sum_regularizer', [weights]) as name:
            my_scale = ops.convert_to_tensor(scale,
                                             dtype=weights.dtype.base_dtype,
                                             name='scale')
            return standard_ops.multiply(
                my_scale,
                standard_ops.reduce_sum(weights),
                name=name)

    return sum_reg