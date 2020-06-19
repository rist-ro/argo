import tensorflow as tf
import sonnet as snt
from sonnet.python.modules.basic import Linear
import os

class LinearWN(Linear):
    
    def __init__(self, *args, use_weight_norm=True, name="linear_wn", **kwargs):
        """
        Args:
            args and kwargs: all the additional keyword arguments will be passed to snt.Linear layers.
        """
        
        custom_getter = None
        
        if use_weight_norm:
            initializer_g = kwargs['initializers']['g']
            initializer_v = kwargs['initializers']['v']
            
            def custom_getter_w(getter, name, *args, **kwargs):
                shape = kwargs["shape"]
                dirname = os.path.dirname(name)
                
                #v = getter(dirname+"/v", *args, **kwargs)
                # new way to create the variable, so that I can specify the initializer
                v = tf.get_variable("v", initializer = initializer_v([shape]))

                output_size = shape[-1]
                initial_value_g = initializer_g([output_size]) * 2
                ln_g = tf.get_variable("ln_g", initializer = initial_value_g)
                # use weight normalization (Salimans & Kingma, 2016)
                w = tf.reshape(tf.exp(ln_g), [1, output_size]) * tf.nn.l2_normalize(v, [0])
                return w
            
            custom_getter = {"w" : custom_getter_w}

        # remove keys from the dictionary, since sonnet check for this
        kwargs['initializers'].pop('g', None)
        kwargs['initializers'].pop('v', None)
        
        super().__init__(*args, **kwargs,
                                        custom_getter=custom_getter, name=name)
