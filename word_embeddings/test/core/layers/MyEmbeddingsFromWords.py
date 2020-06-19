import tensorflow as tf
import numpy as np

class MyEmbeddingsFromWords(tf.keras.layers.Layer):
    def __init__(self, dictionary, embeddings):
        super().__init__()
        self._table_dictionary = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(dictionary.keys()),
                values=list(dictionary.values()),
            ),
            default_value=tf.constant(-1),
            name="vocab"
        )

        unk_emb = np.expand_dims(np.zeros(embeddings.shape[1]), axis=0)
        self._all_embs = tf.constant(np.concatenate((embeddings, unk_emb), axis=0), dtype=tf.float32)

    def call(self, inputs):
        words_indexes = self._table_dictionary.lookup(inputs)
        batch_embs = tf.nn.embedding_lookup(self._all_embs, words_indexes)
        return batch_embs

