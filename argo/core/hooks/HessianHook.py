from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook

import os
import math
import numpy as np
import tensorflow as tf

from tensorflow.python.ops.parallel_for.gradients import jacobian
from scipy.linalg import eigvalsh
from argo.core.argoLogging import get_logger

tf_logging = get_logger()


def my_loss_full_logits(y, logits):
    one_hot_y = tf.one_hot(y, 10)
    softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits))
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(one_hot_y * tf.log(softmax), axis=1))

    return cross_entropy


class HessianHook(EveryNEpochsTFModelHook):
    def __init__(self,
                 model,
                 datasets_keys,
                 period,
                 time_reference,
                 record_eigenvalues,
                 block_size_megabytes,
                 dirName):

        self._dirName = dirName + '/hessian'

        super().__init__(model, period, time_reference, dirName=self._dirName)

        self._handle = model.ds_handle
        self._ds_initializers = model.datasets_initializers
        self._ds_handles_nodes = model.datasets_handles_nodes

        self._record_eigenvalues = record_eigenvalues

        # Block size is expressed in number of elements (float64 - hence the division by 8 bytes)
        self._block_size = int(block_size_megabytes * 1024 * 1024) // 8

        self._period = period
        self._datasets_keys = datasets_keys
        self._hook_name = "hessian_hook"
        tf_logging.info("Create HessianHook for: " + ", ".join(datasets_keys))

    def _begin_once(self):

        with tf.variable_scope('HessianHook'):
            # Define two temp groups: one for storing past values for variables and one for storing current ones.
            temp_group_past = {}
            temp_group_current = {}
            for var in tf.trainable_variables():
                temp_group_past[var.name] = tf.Variable(initial_value=var, trainable=False)
                temp_group_current[var.name] = tf.Variable(initial_value=var, trainable=False)

            # Define 4 operations: two for storing variable values into temp, and two for loading from temps.
            # OP1. Store to 'past' register.
            self._store_to_past_op = tf.group([tf.assign(temp_group_past[var.name], var)
                                               for var in tf.trainable_variables()])

            # OP2. Load from 'past' register.
            self._load_from_past_op = tf.group([tf.assign(var, temp_group_past[var.name])
                                                for var in tf.trainable_variables()])

            # OP3. Store to 'current' register.
            self._store_to_current_op = tf.group([tf.assign(temp_group_current[var.name], var)
                                                  for var in tf.trainable_variables()])

            # OP4. Load from 'current' register.
            self._load_from_current_op = tf.group([tf.assign(var, temp_group_current[var.name])
                                                   for var in tf.trainable_variables()])

            # For debug purposes, define an op that evaluates all variables.
            self._eval_vars = tf.trainable_variables()
            self._eval_temp_current = [temp_group_current[var.name] for var in tf.trainable_variables()]
            self._eval_temp_past = [temp_group_past[var.name] for var in tf.trainable_variables()]

            # TODO: currently, the loss does not depend on the ACTUAL loss of the model. This needs to be fixed.
            loss = my_loss_full_logits(self._model.raw_y, self._model.logits)

            self.hessian_gatherer = HessianGatherer(y=loss,
                                                    var_list=tf.trainable_variables(),
                                                    block_size=self._block_size)

        self._nodes_to_be_computed_by_run["x"] = self._model.raw_x
        self._nodes_to_be_computed_by_run["y"] = self._model.raw_y
        # This will not return anything, but it will store the weights of the model in the temp register.
        self._nodes_to_be_computed_by_run["store_to_past"] = self._store_to_past_op

        # Debug goodies.
        # H = run_values.results["H"]
        # session = run_context.session
        # x = run_values.results["x"]
        # y = run_values.results["y"]
        # H_bis = session.run(self.H, feed_dict = {self._model.raw_x : x, self._model.raw_y : y})

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)
        self._ds_handles = session.run(self._ds_handles_nodes)

    def do_when_triggered(self, global_step, time_ref, run_context, run_values, time_ref_str="ep"):
        tf_logging.info("trigger for HessianHook")

        # Get values relevant for hessian computation.
        session = run_context.session
        x = run_values.results["x"]
        y = run_values.results["y"]

        # TODO: Maybe move this comment to either the class doc or to the file doc.
        """
        We need to have a pair of (input, weights) that represent the same time point, i.e., both of them are used 
        together to compute a gradient update at some point in time (either 'this' step or the next one). 
        In order to compute the Hessian, we can pick either the (input, weights) pair representing the 
        previous -- most recently computed -- step, or the next one (the one that will be computed in the future).
        We currently have access to the previous inputs by registering them in the run op, and next step weights since
        they are the result of the previous step update op. It is hard to access the future inputs because of dataset
        iterators, so we chose to access the previous step weights: we do this by registering them in a temporary
        register at sess.run() time and recover them from that temp variable at hessian computation time. For 
        book keeping, we also need to store the current weight values to another temp variable, since we need to restore
        them when we are done computing the Hessian.
        """

        # Store the current weight values in a temp register:
        session.run(self._store_to_current_op)

        # Load the weight values as they were before the grad update.
        session.run(self._load_from_past_op)

        # Compute hessian on previous value of variables using previous inputs.
        hess = self.hessian_gatherer.get_hessian(sess=session, feed_dict={self._model.raw_x: x, self._model.raw_y: y})

        # Restore the weight values as they were after the most recent grad update.
        session.run(self._load_from_current_op)

        # Save the hessian to a .npy file.
        hess_file_name = 'hessian_' + time_ref_str + "_" + str(time_ref).zfill(4)
        np.save(os.path.join(self._dirName, hess_file_name), hess)

        # Save the eigenvalues if requested.
        if self._record_eigenvalues:
            lambdas = eigvalsh(hess)

            lambdas_file_name = 'eigenvalues_' + time_ref_str + "_" + str(time_ref).zfill(4)
            np.save(os.path.join(self._dirName, lambdas_file_name), lambdas)

        # We should use this only in the case of a full-blown hessian across the entire dataset (train and test).
        # for ds_key in self._datasets_keys:
        #     # images = self._images[ds_key][1]
        #
        #     session = run_context.session
        #     dataset_initializer = self._ds_initializers[ds_key]


class HessianGatherer:
    """
    Defines an object that will compute the (flattened) Hessian matrix with respect to a set of tf graph variables.
    The Hessian is computed in blocks that fit in the GPU memory and further assembled on the CPU.
    """

    def __init__(self, y, var_list, block_size):
        """
        Initialize the hessian gatherer object.

        :param y:                   The value to differentiate. If a rank 1 tensor is specified, a rank 3 tensor will be
                                    computed instead of a rank 2 one. The GPU computation will proceed by evaluating the
                                        hessian for all of those values at once - adjust the block size accordingly!
        :param var_list:            List of tf.Variable objects for which to compute the Hessian.
        :param block_size:          The hessian is evaluated on the GPU in blocks - this specifies the (maximum) number
                                        of elements in these blocks.
        """

        self._var_list = var_list
        self._build_slice_map()

        self._block_size = block_size

        # Build the computational graph.
        self._op_grid = self._build_hessian_op_grid(y)

    def get_hessian(self, sess, feed_dict):
        """
        Computes the hessian block-wise using pairs of vars and returns it as a numpy array.

        :param sess:        Tensorflow session.
        :param feed_dict:   Feed dict to pass on to sess.run()
        :return:            ndarray of hessian.
        """

        # Redefining for readability
        full_width = self._full_width

        # The output hessian in its full width and glory.
        full_hessian = np.zeros(shape=(full_width, full_width))

        # Iterate over pairs of variables and compute the chunk of the Hessian that represents the second derivative
        #   with respect to all of the elements of the two variables.
        for i in range(len(self._var_list)):
            i_var = self._var_list[i]
            i_start_ix, i_end_ix = self._slice_map[i_var.name]

            for j in range(i, len(self._var_list)):
                j_var = self._var_list[j]
                j_start_ix, j_end_ix = self._slice_map[j_var.name]

                # Gather the hessian block representing the derivatives wrt the variable pair.
                pairwise_hess_gatherer = self._op_grid[i][j]
                hess_block = pairwise_hess_gatherer.get_hessian(sess, feed_dict)

                # Assign the hessian block to its corresponding place in the hessian (takes into account symmetry).
                full_hessian[i_start_ix:i_end_ix, j_start_ix:j_end_ix] = hess_block
                full_hessian[j_start_ix:j_end_ix, i_start_ix:i_end_ix] = hess_block.T

        return full_hessian

    def _build_hessian_op_grid(self, y):
        n_vars = len(self._var_list)

        # Build a grid(i,j) of VariablePairHessianGatherer objects that represent the Hessian chunk wrt
        #   elements of Tensorflow variables i and j.
        pairwise_hess_grid = [x[:] for x in [[None] * n_vars] * n_vars]
        for i in range(n_vars):
            for j in range(i, n_vars):
                pairwise_hess_grid[i][j] = VariablePairHessianGatherer(y=y,
                                                                       wrt_x1=self._var_list[i],
                                                                       wrt_x2=self._var_list[j],
                                                                       block_size=self._block_size)

        return pairwise_hess_grid

    def _build_slice_map(self):
        # We are building a dict that maps a variable name to its slice indices (start, end) in a virtual flat array
        #   that contains all var elements.
        current_ix = 0

        slice_map = {}
        for var in self._var_list:
            var_size = int(np.prod(var.shape))
            slice_map[var.name] = (current_ix, current_ix + var_size)

            # Update the current index.
            current_ix += var_size

        self._slice_map = slice_map
        self._full_width = current_ix

    def get_var_slice_ixs(self, var_name):
        return self._slice_map[var_name]


class VariablePairHessianGatherer:
    """
    Utility class used to compute the chunk of Hessian that represents the second derivative of a function wrt the
        elements of two variables. The computation is split into blocks that fit on the GPU and assembled on the CPU.
    """

    def __init__(self, y, wrt_x1, wrt_x2, block_size):
        """
        Initialize the pairwise Hessian Gatherer.

        :param y:               The value to differentiate. If a rank 1 tensor is specified, a rank 3 tensor will be
                                    computed instead of a rank 2 one. The GPU computation will proceed by evaluating the
                                    hessian for all of those values at once - adjust the block width accordingly!
        :param wrt_x1:          The first tensor with respect to which to compute the hessian.
        :param wrt_x2:          The second tensor with respect to which to compute the hessian.
        :param block_size:      The hessian is evaluated on the GPU in blocks - this specifies the (maximum) number of
                                    elements of these blocks.
        """

        # Register some useful numbers.
        self._full_height = int(np.prod(wrt_x1.shape))
        self._full_width = int(np.prod(wrt_x2.shape))
        self._block_height = int(block_size / self._full_width)

        if self._block_height == 0:
            raise ValueError(
                'The block size set for the Hessian Hook is too small at {} elements ({:.2f} MB). '
                'Needs at least {} elements ({:.2f} MB).'.format(block_size, (block_size * 8) / 1048576,
                                                                 self._full_width, (self._full_width * 8) / 1048576))

        self._grid_height = math.ceil(self._full_height / self._block_height)

        # Build the computational graph.
        self._op_grid = self._build_hessian_op_grid(y, wrt_x1, wrt_x2)

    def _build_hessian_op_grid(self, y, wrt_x1, wrt_x2):
        # DEBUG: self.monolith_hessian = jacobian(tf.gradients(y, wrt_x1)[0], wrt_x2, use_pfor=False)
        # DEBUG: self.monolith_hessian = tf.reshape(self.monolith_hessian, shape=(self._full_height, self._full_width))

        # Compute the full gradient wrt x1. We will slice this later when computing the hessian in a blocky manner.
        full_gradient = tf.gradients(y, wrt_x1)[0]
        full_gradient = tf.reshape(full_gradient, shape=(self._full_height,))

        # Redefining for readability
        grid_height = self._grid_height
        block_height = self._block_height
        full_height = self._full_height

        # Build a grid of Tensorflow operations - each operation computes a chunk of the Hessian.
        #   The grid is actually just a list of chunks of shape [block_height, n_elements(x2)].
        #   When assembled, the hessian chunk will have shape [n_elements(x1), n_elements(x2)].
        op_grid = [None] * grid_height
        for i_block in range(grid_height):
            # Parameters for the j-axis of the hessian.
            i_start_ix = i_block * block_height
            i_end_ix = min(full_height, (i_block + 1) * block_height)

            # Add to output op grid.
            grad_chunk = full_gradient[i_start_ix:i_end_ix]
            # TODO: it is not clear whether to use pfor or not (since it is still experimental - maybe we can test).
            #  See https://github.com/tensorflow/tensorflow/issues/675#issuecomment-404665051
            hess_chunk = jacobian(grad_chunk, wrt_x2, use_pfor=False)
            hess_chunk = tf.reshape(hess_chunk, shape=(grad_chunk.shape[0], self._full_width))

            op_grid[i_block] = hess_chunk

        return op_grid

    def get_hessian(self, sess, feed_dict):
        """
        Computes the hessian block-wise and returns it as a numpy array.

        :param sess:        Tensorflow session.
        :param feed_dict:   Feed dict to pass on to sess.run()
        :return:            ndarray of hessian.
        """

        # Redefining for readability
        block_height = self._block_height
        full_width = self._full_width
        full_height = self._full_height

        # The output hessian in its full width and glory.
        full_hessian = np.zeros(shape=(full_height, full_width))

        for i, op in enumerate(self._op_grid):
            i_start_ix = i * block_height
            i_end_ix = min(full_height, (i + 1) * block_height)

            hess_block = sess.run(op, feed_dict)
            full_hessian[i_start_ix:i_end_ix, :] = hess_block

        return full_hessian
