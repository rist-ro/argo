import tensorflow as tf
import sonnet as snt
from sonnet.python.modules.conv import Conv2DTranspose
import os

class Conv2DTransposeWN(Conv2DTranspose):

    def __init__(self, *args, use_weight_norm=True, name="conv2d_transpose_wn", **kwargs):
        """
        Args:
            args and kwargs: all the additional keyword arguments will be passed to snt.Conv2D layers.
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
                
                output_channels = shape[-1]
                initial_value_g = initializer_g([output_channels]) * 2
                ln_g = tf.get_variable("ln_g", initializer = initial_value_g)
                # use weight normalization (Salimans & Kingma, 2016)
                w = tf.reshape(tf.exp(ln_g), [1, 1, 1, output_channels]) * tf.nn.l2_normalize(v, [0, 1, 2])
                return w
            
            custom_getter = {"w" : custom_getter_w}
            
        # remove keys from the dictionary, since sonnet check for this
        kwargs['initializers'].pop('g', None)
        kwargs['initializers'].pop('v', None)
        
        super().__init__(*args,
                                **kwargs,
                                custom_getter=custom_getter, name=name)



#
#     with tf.variable_scope(name):
#         if mode == "init":
#             #print("In conv in if mode==init")
#             # pdb.set_trace()
#             # data based initialization of parameters
#             v = tf.get_variable("V", filter_shape, dtype, tf.random_normal_initializer(0, 0.05, dtype=dtype))
#             v = v.initialized_value()
#             if mask is not None:  # Used for auto-regressive convolutions.
#                 v = mask * v
#
#             v_norm = tf.nn.l2_normalize(v, [0, 1, 2])
#
#             x_init = tf.nn.convolution(x, v_norm, padding, stride, data_format="NHWC")
#
#             m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
#             scale_init = init_scale / tf.sqrt(v_init + 1e-10)
#
#             initial_values_b = -m_init * scale_init
#             initial_values_g = tf.log(scale_init) / 3.0
#
#             local_session = tf.Session()
#             initial_values_b_np, initial_values_g_np = local_session.run([initial_values_b,initial_values_g])
#
#             g = tf.get_variable("g", output_channels,
#                                 initializer=tf.constant_initializer(initial_values_g_np, dtype=dtype))
#
#             b = tf.get_variable("b", output_channels,
#                                      initializer=tf.constant_initializer(initial_values_b_np, dtype=dtype))
#
#             local_session.close()
#
#             return tf.reshape(scale_init, [1, 1, 1, -1]) * (x_init - tf.reshape(m_init, [1, 1, 1, -1]))
#
#
#
# v = tf.get_variable("V", filter_shape)
# g = tf.get_variable("g", [output_channels])
# b = tf.get_variable("b", [output_channels])
# if mask is not None:
#     v = mask * v
#
# # use weight normalization (Salimans & Kingma, 2016)
# w = tf.reshape(tf.exp(g), [1, 1, 1, output_channels]) * tf.nn.l2_normalize(v, [0, 1, 2])
#
# # calculate convolutional layer output
# b = tf.reshape(b, [1, 1, 1, -1])
#
# # pdb.set_trace()
# # b = tf.transpose(b, (0, 2, 3, 1))  # go to NHWC data layout
# # x = tf.transpose(x, (0, 2, 3, 1))  # go to NHWC data layout
#
# # r = tf.nn.conv2d(x, w, stride_shape, padding, data_format="NHWC") + b
#
# # conv_sonnet = snt.Conv2D(name=name,
# #                          output_channels=output_channels,
# #                          kernel_shape=filter_size[0],
# #                          stride=stride[0],
# #                          padding=padding)
# # conv_sonnet._w = w
# # conv_sonnet._b = b
# # r = conv_sonnet(x)
# # # return tf.transpose(r, (0, 3, 1, 2))
#
# r = tf.nn.convolution(x, w, padding, stride, data_format="NHWC") + b
