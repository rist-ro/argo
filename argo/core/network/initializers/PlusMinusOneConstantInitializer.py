import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.init_ops import Initializer, _assert_float_dtype, _compute_fans


class PlusMinusOneConstantInitializer(Initializer):
    """PlusMinusOneConstantInitializer

    Args:
    w,b: the constant inits for w and b
    """

    def __init__(self,
                 w, b,
                 dtype=dtypes.float32):

        self.w = w
        self.b = b
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        scale_shape = shape
        if partition_info is not None:
            scale_shape = partition_info.full_shape
        fan_in, fan_out = _compute_fans(scale_shape)
        return - tf.ones(shape, dtype=dtype) * fan_in * self.w / 2.0 + self.b

    def get_config(self):
        return {
            "w":     self.w,
            "b":     self.b,
            "dtype": self.dtype.name
        }
