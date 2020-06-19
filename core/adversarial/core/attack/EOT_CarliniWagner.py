import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

class EOT_CarliniWagner:
    def __init__(self, sess, defend, model, epsilon, proj_ord=2, ldist_ord=2, data_interval=[0,1], batch_size=50, sample_size = 30, num_steps = 1000, learning_rate = 0.1, const = 10, k = 0, debug = False, input_shape = [28,28,1], n_classes = 10):
        self._sess = sess

        self.epsilon = epsilon
        self._batch_size = batch_size
        self.input_shape = input_shape

        self._input = tf.placeholder(tf.float32, (None,) + input_shape)
        self._x = x = tf.Variable(np.zeros((self._batch_size,) + input_shape), dtype=np.float32, trainable=False)
        self.assign_input_to_x = tf.assign(x, self._input)

        x_expanded = tf.expand_dims(x, axis=1)
        x_repeated = tf.tile(x_expanded, (1, sample_size, 1, 1, 1))

        self.delta = delta = tf.Variable(np.zeros((self._batch_size,) + input_shape), dtype=np.float32)

        if proj_ord==np.inf:
            self.delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)
        elif proj_ord==0:
            self.delta = delta
        elif proj_ord==2:
            self.delta = self.epsilon * tf.nn.l2_normalize(delta)
        else:
            raise Exception("not yet implemented proj_ord=%d"%proj_ord)

        delta_expanded = tf.expand_dims(self.delta, axis = 1)
        delta_repeated = tf.tile(delta_expanded, (1, sample_size, 1, 1, 1))

        self.x_adv = tf.clip_by_value(self._x + self.delta, data_interval[0], data_interval[1])

        x_repeated_clipped = tf.clip_by_value(x_repeated + delta_repeated, data_interval[0], data_interval[1])
        ensemble_xs = tf.map_fn(defend, x_repeated_clipped)
        self._logits = logits = tf.map_fn(model, ensemble_xs)
        self._preds = tf.argmax(self._logits, axis = 2)

        self._label = tf.placeholder(tf.int32, (None,))
        self._y = y = tf.Variable(np.zeros((self._batch_size,)), dtype=tf.int32, trainable=False)
        self.assign_label_to_y = tf.assign(y, self._label)

        one_hot = tf.expand_dims(tf.one_hot(y, n_classes), axis = 1)
        ensemble_labels = tf.tile(one_hot, (1, sample_size, 1))

        correct_logit = tf.reduce_sum(tf.multiply(ensemble_labels, self._logits), axis = 1)
        wrong_logit = tf.reduce_max(tf.multiply(1 - ensemble_labels, self._logits) - 1e4 * ensemble_labels, axis = 1)

        self.loss_adv = tf.reduce_mean(tf.maximum(correct_logit - wrong_logit, -k))
        self.loss = tf.norm(self.delta) + const * self.loss_adv

        #All variables in the graph before I instantiate the optimizer
        start_vars = set(n.name for n in tf.global_variables())

        optimizer = tf.train.AdamOptimizer(learning_rate)
        grad,var = optimizer.compute_gradients(self.loss, [self.delta])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])

        #All variables in the graph after I instantiate the optimizer
        end_vars = tf.global_variables()
        self.optimizer_vars = [n for n in end_vars if n.name not in start_vars]

        self._epsilon = epsilon
        self._max_steps = num_steps
        self._learning_rate = learning_rate
        self._debug = debug

    def run(self, x, y):

        current_batch_size = x.shape[0]

        if current_batch_size < self._batch_size:
            padded_x = np.zeros((self._batch_size,) + self.input_shape)
            padded_x[0:x.shape[0]] = x
            x = padded_x
            padded_y = np.zeros((self._batch_size,))
            padded_y[:y.shape[0]] = y
            y = padded_y

        self._sess.run(tf.variables_initializer(self.optimizer_vars))
        self._sess.run(self.delta.initializer)
        self._sess.run(self.assign_input_to_x, feed_dict={self._input : x})
        self._sess.run(self.assign_label_to_y, feed_dict={self._label : y})

        for i in range(self._max_steps):
            self._sess.run(self.train)

        p, adv = self._sess.run([self._preds, self.x_adv])

        return adv[:current_batch_size]
