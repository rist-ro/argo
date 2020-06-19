import tensorflow as tf
import sonnet as snt

from .AbstractModule import AbstractModule


class Dropout(AbstractModule):

    #<<<<<<< HEAD
    #def __init__(self, is_training, noise_shape=None, seed=None, rate=None, name="Dropout"):
    #=======
    def __init__(self, rate, dropout_flag=None, name="Dropout"):
        #>>>>>>> e9f208ac74d05b1259d1b1d63445a54687bae812
        super().__init__(name = name)
        self._rate = rate
        self._dropout_flag = dropout_flag
        self._name = name

        #<<<<<<< HEAD
        #self._is_training = is_training

    '''
    def _build(self, input):
        return tf.nn.dropout(input,
                             noise_shape = self._noise_shape,
                             seed = self._seed,
                             name = self._name,
                             keep_prob = self._rate
        )

        #return tf.layers.dropout(input,
        #                         rate=self._rate,
        #                         noise_shape=self._noise_shape,
        #                         seed=self._seed,
        #                         training=self._is_training,
        #                         name=self._name
        #)
    
    
        '#return tf.keras.layers.Dropout(input,
         #                              rate=self._rate,
         #                              noise_shape=self._noise_shape,
         #                              seed=self._seed,
         #                              training=self._is_training,
         #                              name=self._name)
    
    '''
    #=======
    def _build(self, inputs):
        return tf.layers.dropout(inputs, self._rate,
                                 training = self._dropout_flag,
                                 name = self._name)

    # def __init__(self, noise_shape=None, seed=None, rate=None, name="Dropout"):
    #
    #     super().__init__(name = name)
    #
    #     self._noise_shape = noise_shape
    #     self._seed = seed
    #     self._rate = rate
    #
    # def _build(self, input):
    #     return tf.nn.dropout(input,
    #                          noise_shape = self._noise_shape,
    #                          keep_prob = self._rate,
    #                          seed = self._seed,
    #                          name = self.module_name,
    #     )

