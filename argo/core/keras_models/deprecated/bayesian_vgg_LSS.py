import tensorflow as tf
import tensorflow_probability as tfp
from .keras_utils import get_renorm_clipping

def bayesian_vgg_LSS(filters = [32, 64, 128, 256],#[32, 64, 64, 128, 128],
                 kernels = [3, 3, 3, 3],
                 strides = [2, 2, 2, 2],
                 logits_size = None,
                 flipout = True,
                 pooling = "max",
                 renorm = False,
                 layer_kwargs = {},
                 layer_kwargs_bayes = {}):

    """Constructs a VGG16 model.

    Args:
    input_shape: A `tuple` indicating the Tensor shape.
    num_classes: `int` representing the number of class labels.
    kernel_posterior_scale_mean: Python `int` number for the kernel
      posterior's scale (log variance) mean. The smaller the mean the closer
      is the initialization to a deterministic network.
    kernel_posterior_scale_stddev: Python `float` number for the initial kernel
      posterior's scale stddev.
      ```
      q(W|x) ~ N(mu, var),
      log_var ~ N(kernel_posterior_scale_mean, kernel_posterior_scale_stddev)
      ````
    kernel_posterior_scale_constraint: Python `float` number for the log value
      to constrain the log variance throughout training.
      i.e. log_var <= log(kernel_posterior_scale_constraint).

    Returns:
    tf.keras.Model.
    """

    pooling_choices = ["max", "avg", None]
    if pooling not in pooling_choices:
        raise ValueError("pooling must be in {:}, instead {:} found.".format(pooling_choices, pooling))

    if flipout:
        Conv2D = tfp.layers.Convolution2DFlipout
        Dense = tfp.layers.DenseFlipout
    else:
        Conv2D = tfp.layers.Convolution2DReparameterization
        Dense = tfp.layers.DenseReparameterization

    renorm_clipping = None
    if renorm:
        renorm_clipping = get_renorm_clipping()

    layers_list = []
    layers_list+=Blocks(Conv2D, filters[0], kernels[0],  renorm,renorm_clipping,2,**layer_kwargs_bayes)
    layers_list+=pool(pooling, strides[0])
    layers_list+=Blocks(Conv2D, filters[1], kernels[1],  renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[0], 1, renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[1], kernels[1], renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=pool(pooling, strides[1])
    layers_list+=Blocks(Conv2D, filters[2], kernels[2], renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[1], 1, renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[2], kernels[2], renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=pool(pooling, strides[2])
    layers_list+=Blocks(Conv2D, filters[3], kernels[3],  renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[2], 1, renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[3], kernels[3], renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=pool(pooling, strides[3])
    layers_list+=Blocks(Conv2D, filters[3], kernels[3], renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[2], 1,  renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[3], kernels[3], renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[2], 1,  renorm,renorm_clipping,1,**layer_kwargs_bayes)
    layers_list+=Blocks(Conv2D, filters[3], kernels[3], renorm,renorm_clipping,1,bn_last=False,**layer_kwargs_bayes)
    layers_list+=pool(pooling, strides[3])

    # if logits_size is specified I add an extra Dense layer
    if logits_size is not None:
        layers_list += [tf.keras.layers.Flatten(),
                        Dense(
                            logits_size,
                            **layer_kwargs_bayes)]
    model = tf.keras.Sequential(layers_list, name='vgg')
    return model


# def _vggconv_block(Conv2D, filters, kernel, stride, pooling, renorm, renorm_clipping,**layer_kwargs_bayes):
#     """Network block for VGG."""
#
#     layers_list = []
#     layers_list+=Blocks(Conv2D, filters, kernel,  renorm,renorm_clipping,2,**layer_kwargs_bayes)
#     layers_list+=pool(pooling, stride)
#     layers_list+=Blocks(Conv2D, filter, kernel,  renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, 1, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, filter, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=pool(pooling, stride)
#     layers_list+=Blocks(Conv2D, filter, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, 1, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, filter, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=pool(pooling, stride)
#     layers_list+=Blocks(Conv2D, filter, kernel,  renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, 1, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, filter, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=pool(pooling, stride)
#     layers_list+=Blocks(Conv2D, filter, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, 1, kernel,  renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, filter, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, 1, kernel,  renorm,renorm_clipping,1,**layer_kwargs_bayes)
#     layers_list+=Blocks(Conv2D, filter, kernel, renorm,renorm_clipping,1,**layer_kwargs_bayes)



    #
    #
    # conv_stride = get_conv_stride(pooling, stride)
    #
    # layers_list+= [
    #     Conv2D(
    #         filters,
    #         kernel,
    #         strides=conv_stride,
    #         padding='same',
    #         **layer_kwargs_bayes),
    #
    #     tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping),
    #
    #     tf.keras.layers.LeakyReLU()]
def Blocks(Conv2D, filters, kernel,  renorm,renorm_clipping,n_times,bn_last=True,**layer_kwargs_bayes):
    layers_list =[
            Conv2D(
                filters,
                kernel,
                padding='valid',
                **layer_kwargs_bayes)]
    if bn_last:
        layers_list +=[tf.keras.layers.BatchNormalization(renorm=renorm, renorm_clipping=renorm_clipping)]
    else:
        pass
    layers_list +=[tf.keras.layers.LeakyReLU()]
    return layers_list*n_times


def pool(pooling, stride):
    if pooling=="max":
            layers_list =[tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=stride)]
    elif pooling=="avg":
            layers_list = [tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=stride)]
    return layers_list


#def get_conv_stride(pooling, stride):

#    if pooling is None:
#        conv_stride = stride
#    else:
#        conv_stride = 1
#    return conv_stride
