import tensorflow as tf
import numpy as np

class CarliniWagner:
    def __init__(self, sess, model, num_steps, proj_ord=np.inf, epsilon=0.1, batch_size=50, ldist_ord=2, const=1, data_interval = [0., 1.],
                 learning_rate=0.1, input_shape=[28, 28, 1], n_classes=10):
        """
        Args:
            sess (tf.Session): the tf session
            model: the function taking an input x and creating the graph nodes and returning the logits of the model
            num_steps: the number of steps to do
            proj_ord: the order of the projection (0: no projection, int>0: project according to the norm ord=int
                                                                    make norm of the perturbation equal to epsilon,
                                                np.inf: project in the box with sides epsilon)
            epsilon: the epsilon used for the projection (unless prj_ord=0, then no projection is done)
            batch_size: how many examples are processed at a time
            const: constant to weight the adv loss against the ldistance loss
            ord: the ord of the norm used for the ldistance in the CW variational problem
            data_interval: the interval of the data
            learning_rate: learning rate for the adam optimizer
            input_shape: the input shape
            n_classes: the number of classes for the classification
        """

        self._sess = sess
        self._model = model

        self.num_steps = num_steps

        self.batch_size = batch_size
        self._curr_bs = self.batch_size

        self.input_shape = input_shape

        self._x = x = tf.Variable(np.zeros((batch_size,) + input_shape, dtype=np.float32),
                                            name='modifier')
        self._input = tf.placeholder(tf.float32, (None,) + input_shape)
        self._input_xvar = input_xvar = tf.Variable(np.zeros((batch_size,) + input_shape, dtype=np.float32),
                                                    name='input_x')
        self._assign_input_to_var = tf.assign(self._input_xvar, self._input)

        self._label = tf.placeholder(tf.int32, (None,))
        self._y = y = tf.Variable(np.zeros((batch_size,)), dtype=tf.int32, trainable=False)
        self._assign_label_to_y = tf.assign(self._y, self._label)

        self.epsilon = epsilon

        delta = tf.clip_by_value(self._x, data_interval[0], data_interval[1]) - input_xvar
        # projection of delta
        if proj_ord==np.inf:
            delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)
        elif proj_ord==0:
            delta = delta
        elif proj_ord==2:
            delta = self.epsilon * tf.nn.l2_normalize(delta)
        else:
            raise Exception("not yet implemented proj_ord=%d"%proj_ord)

        self._assign_xs_and_clip = tf.assign(self._x, input_xvar + delta)
        
        x_expanded = tf.expand_dims(self._x, axis=1)
        self._logits = logits = tf.map_fn(model, x_expanded)

        self._preds = tf.argmax(self._logits, axis=2)

        one_hot = tf.expand_dims(tf.one_hot(self._y, n_classes), axis=1)

        correct_logit = tf.reduce_sum(one_hot * logits, axis=2)
        wrong_logit = tf.reduce_max((1-one_hot) * logits - 1e4*one_hot, axis=2)

        ldist = 0.
        if ldist_ord>0:
            if self.batch_size == 1:
                ldist = tf.norm(self._x - input_xvar, ord=ldist_ord)
            else:
                ldist = tf.norm(self._x - input_xvar, ord=ldist_ord, axis=(1,2))
                
        self._loss2 = ldist
        self._loss1 = correct_logit - wrong_logit
        self._loss_arr = const*self._loss1 + self._loss2
        self._loss = tf.reduce_sum(self._loss_arr)

        # All variables in the graph before I instantiate the optimizer
        start_vars = set(n.name for n in tf.global_variables())

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self._train = optimizer.minimize(self._loss, var_list=[self._x])

        # All variables in the graph after I instantiate the optimizer
        end_vars = tf.global_variables()
        self._optimizer_vars = [n for n in end_vars if n.name not in start_vars]

    def run(self, x, y):
        self._sess.run(tf.variables_initializer(self._optimizer_vars))

        current_batch_size = x.shape[0]

        # if we have less than self.batch_size examples left we pad x with zeroes to match self.batch_size
        if current_batch_size < self.batch_size:
            padded_x = np.zeros((self.batch_size,) + self.input_shape)
            padded_x[0:x.shape[0]] = x
            x = padded_x
            padded_y = np.zeros((self.batch_size,))
            padded_y[:y.shape[0]] = y
            y = padded_y
        
        self._sess.run(self._assign_input_to_var, {self._input: x})
        self._sess.run(self._assign_label_to_y, {self._label: y})
        self._sess.run(self._x.initializer)
        self._sess.run(self._assign_xs_and_clip)

        for i in range(self.num_steps):
            _, p, loss = self._sess.run([self._train, self._preds, self._loss_arr])

            self._sess.run(self._assign_xs_and_clip)

        return self._sess.run(self._x[:current_batch_size])
