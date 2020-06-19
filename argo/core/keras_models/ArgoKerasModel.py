from abc import ABC, abstractmethod
import tensorflow as tf

class ArgoKerasModel(tf.keras.Model):

    @abstractmethod
    def _id(self):
        pass

