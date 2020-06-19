import tensorflow as tf
from .keras_utils import parse_layer

def name_net(output_size, keras_layers_list, layer_kwargs = {}, layer_kwargs_bayes = {}, name=""):
    model = tf.keras.Sequential([])
    last_layer_kwargs = keras_layers_list[-1][1]
    last_layer_kwargs.update({"units" : output_size})

    for layer_tuple in keras_layers_list:
        maybe_update(layer_tuple, layer_kwargs)
        layer = parse_layer(layer_tuple)
        model.add(layer)

    return model

def maybe_update(layer_tuple, layer_kwargs):
    layer_name = layer_tuple[0].split(".")[-1]

    if layer_name in ['Dropout', 'Flatten', 'BatchNormalization'] or "Pool" in layer_name:
        return

    layer_tuple[1].update(layer_kwargs)
