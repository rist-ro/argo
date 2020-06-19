import tensorflow as tf

from ..utils.argo_utils import NUMTOL


def my_loss_full_logits(y, logits):

    n = logits.get_shape().as_list()[1]
    probabilities = tf.nn.softmax(logits)

    clipped_probabilities = tf.clip_by_value(probabilities, NUMTOL, 1 - NUMTOL)
    loss = tf.reduce_sum(-tf.one_hot(y,depth=n)*tf.log(clipped_probabilities),axis=1)

    return loss

# TO BE REMOVED!
'''
def my_loss(y, logits_add_last_node):

    n = logits_add_last_node.get_shape().as_list()[1]

    probabilities = tf.nn.softmax(logits_add_last_node)
    probabilities_sliced = tf.slice(probabilities, [0, 0], [-1, n-1])
    new_probabilities = tf.concat([probabilities_sliced,
                                   tf.reshape(1 - tf.reduce_sum(probabilities_sliced, axis=1),[-1,1])],
                                  1)

    clipped_probabilies = tf.clip_by_value(new_probabilities, NUMTOL, 1-NUMTOL)
    loss = tf.reduce_sum(-tf.one_hot(y,depth=n)*tf.log(clipped_probabilies),axis=1)

    return loss
'''

