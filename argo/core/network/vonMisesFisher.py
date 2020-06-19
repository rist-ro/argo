import tensorflow as tf
#import sonnet as snt

import math

import numpy as np

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

# needed for MyVonMisesFsher
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util

# ive
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.custom_gradient import custom_gradient

import scipy.special

from .AbstractStochasticLayer import AbstractStochasticLayer

import pdb

import types

from tensorflow.python.ops.custom_gradient import custom_gradient


@custom_gradient
def _bessel_ive(v, z, cache = None):
    """Exponentially scaled modified Bessel function of the first kind."""
    output = array_ops.reshape(script_ops.py_func(
        lambda v, z: np.select(condlist=[v == 0, v == 1],
                               choicelist=[scipy.special.i0e(z, dtype=z.dtype),
                                           scipy.special.i1e(z, dtype=z.dtype)],
                               default=scipy.special.ive(v, z, dtype=z.dtype)), [v, z], z.dtype),
        ops.convert_to_tensor(array_ops.shape(z), dtype=dtypes.int32))

    def grad(dy):
        return None, dy * (_bessel_ive(v - 1, z) - _bessel_ive(v, z) * (v + z) / z)

    return output, grad


def tf_bessel_ive(v, z, cache=None):
    """Computes I_v(z)*exp(-abs(z)) using a recurrence relation, where z > 0."""
    # TODO(b/67497980): Switch to a more numerically faithful implementation.
    z = tf.convert_to_tensor(z)
  
    wrap = lambda result: tf.debugging.check_numerics(result, 'besseli{}'.format(v
                                                                              ))
    #if float(v) >= 2:
    #    raise ValueError(
    #        'Evaluating bessel_i by recurrence becomes imprecise for large v')
    
    cache = cache or {}
    safe_z = tf.where(z > 0, z, tf.ones_like(z))
    if v in cache:
        return tf.check_numerics(wrap(cache[v]), "wrap(cache[v] fails check numerics")
    if v == 0:
        cache[v] = tf.math.bessel_i0e(z)
    elif v == 1:
        cache[v] = tf.math.bessel_i1e(z)
    elif v == 0.5:
        # sinh(x)*exp(-abs(x)), sinh(x) = (e^x - e^{-x}) / 2
        sinhe = lambda x: (tf.exp(x - tf.abs(x)) - tf.exp(-x - tf.abs(x))) / 2
        cache[v] = (
            np.sqrt(2 / np.pi) * sinhe(z) *
            tf.where(z > 0, tf.math.rsqrt(safe_z), tf.ones_like(safe_z)))
    elif v == -0.5:
        # cosh(x)*exp(-abs(x)), cosh(x) = (e^x + e^{-x}) / 2
        coshe = lambda x: (tf.exp(x - tf.abs(x)) + tf.exp(-x - tf.abs(x))) / 2
        cache[v] = (
            np.sqrt(2 / np.pi) * coshe(z) *
            tf.where(z > 0, tf.math.rsqrt(safe_z), tf.ones_like(safe_z)))
    if v <= 1:
        return tf.check_numerics(wrap(cache[v]), "wrap(cache[v] fails check numerics")
    # Recurrence relation:
    cache[v] = (_bessel_ive(v - 2, z, cache) -
                (2 * (v - 1)) * _bessel_ive(v - 1, z, cache) / z)
    return tf.check_numerics(wrap(cache[v]), "wrap(cache[v] fails check numerics")


# see https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/distributions/von_mises_fisher.py
class MyVonMisesFisher(tfd.VonMisesFisher):
    def __init__(self,
                 mean_direction,
                 concentration,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='VonMisesFisher',
                 check_dim = True):

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([mean_direction, concentration],
                                      tf.float32)
            self._mean_direction = tensor_util.convert_nonref_to_tensor(
                mean_direction, name='mean_direction', dtype=dtype)
            self._concentration = tensor_util.convert_nonref_to_tensor(
                concentration, name='concentration', dtype=dtype)

            static_event_dim = tf.compat.dimension_value(
                tensorshape_util.with_rank_at_least(
                    self._mean_direction.shape, 1)[-1])
            if check_dim == True:
                if static_event_dim is not None and static_event_dim > 5:
                    raise ValueError('von Mises-Fisher ndims > 5 is not currently '
                                     'supported')

            # mean_direction is always reparameterized.
            # concentration is only for event_dim==3, via an inversion sampler.
            reparameterization_type = (
                reparameterization.FULLY_REPARAMETERIZED
                if static_event_dim == 3 else
                reparameterization.NOT_REPARAMETERIZED)
            super(tfd.VonMisesFisher, self).__init__(
                dtype=self._concentration.dtype,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                reparameterization_type=reparameterization_type,
                parameters=parameters,
                name=name)

    def _mean(self):
        # Derivation: https://sachinruk.github.io/blog/von-Mises-Fisher/
        concentration = tf.convert_to_tensor(self.concentration)
        mean_direction = tf.convert_to_tensor(self.mean_direction)

        event_dim = tf.compat.dimension_value(self.event_shape[0])
        if event_dim is None:
            raise ValueError('event shape must be statically known for _bessel_ive')
        safe_conc = tf.where(concentration > 0, concentration,
                             tf.ones_like(concentration))
        safe_mean = mean_direction * (
            _bessel_ive(event_dim / 2, safe_conc) /
            _bessel_ive(event_dim / 2 - 1, safe_conc))[..., tf.newaxis]
        return tf.where(
            concentration[..., tf.newaxis] > 0.,
            safe_mean, tf.zeros_like(safe_mean))

    def _covariance(self):
        # Derivation: https://sachinruk.github.io/blog/von-Mises-Fisher/
        event_dim = tf.compat.dimension_value(self.event_shape[0])
        if event_dim is None:
            raise ValueError('event shape must be statically known for _bessel_ive')
        # TODO(b/141142878): Enable this; numerically unstable.
        if event_dim > 2:
            raise NotImplementedError(
                'vMF covariance is numerically unstable for dim>2')
        mean_direction = tf.convert_to_tensor(self.mean_direction)
        concentration = tf.convert_to_tensor(self.concentration)
        safe_conc = tf.where(concentration > 0, concentration,
                             tf.ones_like(concentration))[..., tf.newaxis]
        h = (_bessel_ive(event_dim / 2, safe_conc) /
             _bessel_ive(event_dim / 2 - 1, safe_conc))
        intermediate = (
            tf.matmul(mean_direction[..., :, tf.newaxis],
                      mean_direction[..., tf.newaxis, :]) *
            (1 - event_dim * h / safe_conc - h**2)[..., tf.newaxis])
        cov = tf.linalg.set_diag(
            intermediate,
            tf.linalg.diag_part(intermediate) + (h / safe_conc))
        return tf.where(
            concentration[..., tf.newaxis, tf.newaxis] > 0., cov,
            tf.linalg.eye(event_dim,
                          batch_shape=self._batch_shape_tensor(
                              mean_direction=mean_direction,
                              concentration=concentration)) / event_dim)

    def _log_normalization(self, concentration=None):
        """Computes the log-normalizer of the distribution."""
        if concentration is None:
            concentration = tf.convert_to_tensor(self.concentration)

        #pdb.set_trace()
        event_dim = tf.compat.dimension_value(self.event_shape[0])
        if event_dim is None:
            raise ValueError('von Mises-Fisher _log_normalizer currently only '
                             'supports statically known event shape')
        safe_conc = tf.where(concentration > 0, concentration,
                             tf.ones_like(concentration))
        safe_lognorm = ((event_dim / 2 - 1) * tf.math.log(safe_conc) -
                        (event_dim / 2) * np.log(2 * np.pi) -
                        tf.math.log(_bessel_ive(event_dim / 2 - 1, safe_conc)) -
                        tf.abs(safe_conc))
        log_nsphere_surface_area = (
            np.log(2.) + (event_dim / 2) * np.log(np.pi) -
            tf.math.lgamma(tf.cast(event_dim / 2, self.dtype)))
        return tf.where(concentration > 0, -safe_lognorm,
                        log_nsphere_surface_area*tf.ones_like(concentration)) # Luigi

    
    def _entropy(self):
        mf = int(self.mean_direction.shape[-1])

        #pdb.set_trace()
        
        #return - tf.reshape(self.concentration * _bessel_ive(mf / 2, self.concentration) / _bessel_ive((mf / 2) - 1, self.concentration),
        #                           tf.convert_to_tensor(tf.shape(self.concentration)[:-1])) + self._log_normalization()

        return -self.concentration * _bessel_ive(mf / 2, self.concentration) / _bessel_ive((mf / 2) - 1, self.concentration) + self._log_normalization()


    def _event_shape(self):
        s = tensorshape_util.with_rank_at_least(self.mean_direction.shape, 1)
        return s[-1:]
    
    def _batch_shape(self):
        return tf.broadcast_static_shape(
            tensorshape_util.with_rank_at_least(self.mean_direction.shape, 1)[:-1],
            self.concentration.shape)


    
# this object is the name of the layer
class vonMisesFisher(AbstractStochasticLayer):
    
    def __init__(self, module_tuple = ("Linear", {}),
                 output_size=None,
                 output_shape=None,
                 minimal_concentration=0.01,
                 concentration_parameterization="softplus",
                 #vectorial_covariance = True, 
                 initializers = {},
                 regularizers = {},
                 contractive_regularizer = None,
                 name='von_mises_diagonal'):
        
        super().__init__(module_tuple = module_tuple,
                         output_size = output_size,
                         output_shape = output_shape,
                         minimal_covariance = minimal_concentration, # a little bit improper (Luigi)
                         covariance_parameterization=concentration_parameterization, # a little bit improper (Luigi)
                         vectorial_covariance = True, # since the vMF has only a scalar covariance
                         initializers = initializers,
                         regularizers = regularizers,
                         contractive_regularizer = contractive_regularizer,
                         name = name)
    
    def _build(self, inputs):
        mean, concentration, _ = self.create_mean_n_cov_layers(inputs)
        # normaize the mean
        mean = tf.nn.l2_normalize(mean, axis=-1)
        
        self.set_contractive_regularizer(mean, concentration,
                                         self._contractive_regularizer_inputs,
                                         self._contractive_regularizer_tuple,
                                         self._contractive_collection_network_str) 

        
        #output_distribution = tfd.Normal(loc=mean, scale=scale)
        concetration = tf.reshape(concentration,[-1])
        output_distribution = MyVonMisesFisher(mean, concetration, check_dim=False)
        #output_distribution = tfd.VonMisesFisher(mean, covariance)
        #pdb.set_trace()
        
        # add reconstruction_node method (needed to some sort of mean or median to get reconstructions without sampling)
        def reconstruction_node(self):
            return self.mean()
        
        output_distribution.reconstruction_node = types.MethodType(reconstruction_node, output_distribution)

        def params(self):
            return (self.mean_direction, self.concentration)

        output_distribution.params = types.MethodType(params, output_distribution)

        def default_prior(self, dim):
            zeros = tf.zeros(shape=[dim])
            prior = MyVonMisesFisher(zeros, 0, check_dim=False, name="prior")
            return prior

        output_distribution.default_prior = types.MethodType(default_prior, output_distribution)

        def kl_divergence(self, p):
            # this formula is correct only if p is the prior!
            dim = int(self.mean_direction.shape[-1])
            entropy_prior = math.log(2) + ((dim + 1) / 2) * math.log(math.pi) - tf.math.lgamma(tf.cast((dim + 1) / 2, dtype=self.dtype))
            #return tf.ones_like(self.concentration) + entropy_prior
            return -self.entropy() + entropy_prior

        output_distribution.kl_divergence = types.MethodType(kl_divergence, output_distribution)

        '''
        def _entropy(self):
            mf = int(self.mean_direction.shape[-1])
            return - tf.reshape(self.concentration * _bessel_ive(mf / 2, self.concentration) / _bessel_ive((mf / 2) - 1, self.concentration),
                                tf.convert_to_tensor(tf.shape(self.concentration)[:-1])) + self._log_normalization()
        
        output_distribution._entropy = types.MethodType(_entropy, output_distribution)
        '''
        
        return output_distribution

    @property
    def _contractive_regularizer_filename(self):
        raise Exception("not sure what it is now")
        return ".vonMisesFisherRegularizers"
