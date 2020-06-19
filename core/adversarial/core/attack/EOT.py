import tensorflow as tf
import numpy as np

class EOT:
    def __init__(self, sess, model, num_steps, sample_size=30, proj_ord=np.inf, epsilon=0.1, batch_size=50,
                 learning_rate=0.1, data_interval=[0., 1.], input_shape=[28, 28, 1], n_classes=10):
        """
        Args:
            sess (tf.Session): the tf session
            model: the function taking an input x and creating the graph nodes and returning the logits of the model
            num_steps: the number of steps to do
            proj_ord: the order of the projection (0: no projection, int>0: project according to the norm ord=int
                                                                    make norm of the perturbation equal to epsilon,
                                                np.inf: project in the box with sides epsilon)
            epsilon: the epsilon used for the projection (unless prj_ord=0, then no projection is done)
            const: constant to weight the adv loss against the ldistance loss
            data_interval: the interval of the data
            learning_rate: learning rate for the adam optimizer
            input_shape: the input shape
        """

        self._sess = sess
        # self._defend = defend
        self._model = model

        self._num_steps = num_steps

        self._batch_size = batch_size
        self._curr_bs = self._batch_size

        self._sample_size = sample_size

        self._input_shape = input_shape

        self._epsilon = epsilon

        self._data_interval = data_interval

        # clean input variable
        self._x = x = tf.Variable(np.zeros((batch_size,) + self._input_shape, dtype=np.float32),
                                            name='modifier')
        self._input = tf.placeholder(tf.float32, (None,) + self._input_shape)
        
        self._y = y = tf.Variable(np.zeros((self._batch_size,)), dtype=tf.int32, trainable=False)
        self._label = tf.placeholder(tf.int32, (None,))

        # clean input variable repeated
        self._input_var = tf.Variable(np.zeros((self._batch_size,) + input_shape, dtype=np.float32),
                                            name='input_rep')

        self._label_repeated = tf.transpose(tf.tile((self._y,), (self._sample_size,1)))

        # initialization
        self.assign_x_var = tf.assign(self._x, self._input)
        self.assign_input_var = tf.assign(self._input_var, self._input)
        # self.assign_xvar_repeated = tf.assign(self._xvar_repeated, self._x_repeated)
        self.assign_label = tf.assign(self._y, self._label)

        delta = tf.clip_by_value(self._x, data_interval[0], data_interval[1]) - self._input_var

        if proj_ord == 'inf':
            proj_ord = np.inf
        if proj_ord==np.inf:
            delta = tf.clip_by_value(delta, -self._epsilon, self._epsilon)
        elif proj_ord==0:
            delta = delta
        elif proj_ord==2:
            delta = self._epsilon * tf.nn.l2_normalize(delta)
        else:
            raise Exception("not yet implemented proj_ord=%d"%proj_ord)

        self._assign_xs_and_clip = tf.assign(self._x, self._input_var + delta)

                # perturbed variable repeated
        # self._xvar_repeated = xvar_repeated = tf.Variable(np.zeros((self._batch_size, self._sample_size) + input_shape, dtype=np.float32),
        #                                     name='modifier_rep')

        x_expanded = tf.expand_dims(self._x, axis=1)
        self._xvar_repeated = tf.tile(x_expanded, (1, self._sample_size, 1, 1, 1))

        # transforming the input
        # self.ensemble_xs = ensemble_xs = tf.map_fn(defend, self._xvar_repeated)
        # self._logits = logits = tf.map_fn(model, self.ensemble_xs)
        self._logits = logits = tf.map_fn(model, self._xvar_repeated)
        # import ipdb; ipdb.set_trace()
        # self._probs = probs = tf.nn.softmax(self._logits)
        self._preds = tf.argmax(self._logits, axis = 2)

        
        # one_hot = tf.one_hot(self._label_repeated, n_classes)
        # correct_prob = tf.reduce_sum(one_hot * probs, axis = 2)

        # # import ipdb; ipdb.set_trace()
        # self._loss = tf.reduce_sum(-tf.math.log(correct_prob))
        # # self._loss = -tf.math.log(correct_prob)
        self._softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self._label_repeated)
        self._loss = tf.reduce_sum(tf.reduce_mean(self._softmax, axis=1))

        # ensemble_grad, = tf.gradients(self._loss, self._input)

        # # All variables in the graph before I instantiate the optimizer
        start_vars = set(n.name for n in tf.global_variables())

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        # import ipdb; ipdb.set_trace()
        self._train = optimizer.minimize(-self._loss, var_list=[self._x])   # maximize the loss
        
        # All variables in the graph after I instantiate the optimizer
        end_vars = tf.global_variables()
        self._optimizer_vars = [n for n in end_vars if n.name not in start_vars]

    def run(self, x, y):
        # import ipdb; ipdb.set_trace()
        self._sess.run(tf.variables_initializer(self._optimizer_vars))

        current_batch_size = x.shape[0]

        # if we have less than self._batch_size examples left we pad x with zeroes to match self._batch_size
        if current_batch_size < self._batch_size:
            padded_x = np.zeros((self._batch_size,) + self._input_shape)
            padded_x[0:x.shape[0]] = x
            x = padded_x
            padded_y = np.zeros((self._batch_size,))
            padded_y[:y.shape[0]] = y
            y = padded_y

        
        self._sess.run(self.assign_x_var, {self._input: x})
        self._sess.run(self.assign_input_var, {self._input: x})
        # self._sess.run(self.assign_xvar_repeated, {self._input: x})
        self._sess.run(self.assign_label, {self._label: y})
        self._sess.run(self._x.initializer)
        self._sess.run(self._assign_xs_and_clip)

        # import ipdb; ipdb.set_trace()
        for i in range(self._num_steps):
            _, p, loss = self._sess.run([self._train, self._preds, self._loss])
            # import ipdb; ipdb.set_trace()
            self._sess.run(self._assign_xs_and_clip)

        return self._sess.run(self._x[:current_batch_size])
