import tensorflow as tf
import numpy as np

class MyEmbeddingsFromInts(tf.keras.layers.Layer):
    def __init__(self, embeddings, dtype=tf.float32):
        super().__init__()

        unk_emb = np.expand_dims(np.zeros(embeddings.shape[1]), axis=0)
        self._all_embs = tf.constant(np.concatenate((embeddings, unk_emb), axis=0), dtype=dtype)

    def call(self, inputs):
        batch_embs = tf.nn.embedding_lookup(self._all_embs, inputs)
        return batch_embs

