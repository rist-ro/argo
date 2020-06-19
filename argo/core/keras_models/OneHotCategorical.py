import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from .ArgoKerasModel import ArgoKerasModel

class OneHotCategorical(ArgoKerasModel):
    
    def _id(self):
        _id = 'Cat'
        return _id
    
    def __init__(self, output_size, layer_kwargs = {}, layer_kwargs_bayes = {}):
        super().__init__(name='onehot_categorical')

        Dense = partial(tf.keras.layers.Dense, **layer_kwargs)
        self._flat = tf.keras.layers.Flatten()
        self._output_size = output_size
        self.dense_logits = Dense(output_size)

    def call(self, inputs, training=False, **extra_kwargs):
        inputs = self._flat(inputs)
        logits = self.dense_logits(inputs)

        ouput_params = {"logits": logits}
        distr = tfp.distributions.OneHotCategorical(**ouput_params)

        # hack because keras does not want distr in output... (Riccardo)
        distr.shape = tf.TensorShape(tuple(distr.batch_shape.as_list() + distr.event_shape.as_list()))

        return distr
