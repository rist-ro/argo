import tensorflow as tf

import tensorflow_probability as tfp

import pdb 

# maybe we should change the name, since it transforms the support, and does not change the log_p
class PlusMinusOneMapping(tfp.bijectors.AffineScalar):

   
    def __init__(self,
                 name='affine_scalar',
                 **kwargs):
    
        super().__init__(name='harmonic_bijector', **kwargs) 
   
    
    def _inverse_log_det_jacobian(self, y):
        return tf.zeros([], dtype=y.dtype)
    
    def _forward_log_det_jacobian(self, x):
        return tf.zeros([], dtype=x.dtype)
