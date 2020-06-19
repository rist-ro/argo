'''
                                    DOCUMENTATION:

I have implemented the SSA 2 algorithms, given in:
https://arxiv.org/abs/1904.12952

Nesterov method with constant momentum factor is given in the work of Defazio:
https://arxiv.org/abs/1812.04634
See Table 1, page 3 - 'Modern Momentum' (here: beta is the momentum factor)

The Default values for the 3 hyper-parameters: 'lr', 'k' and 'beta'
1.) 'k = 2.0'
2.) 'lr = 1e-2'

Here, we have 'beta_n' and 'beta_{n+1}' the current and the predicted non-constant momentum terms
Here, we consider 'beta_n' given by the inertial momentum sequence with the general term 'n / (n + alpha)'
Since 'alpha >= 3.0', we set by default 'alpha = 4.0'
'''



# Loading modules
from tensorflow.python.training import optimizer    # Here we have the 'Optimizer' class
from tensorflow.python.framework import ops         # From here we need the function that converts to 'Tensor' object
from tensorflow.python.ops import math_ops          # From here we need mathematical operations for 'Tensor' objects
from tensorflow.python.ops import state_ops         # From here we need 'Operations' on 'Tensor' objects
from tensorflow.python.ops import control_flow_ops  # From here we need the function 'group'




# The subclass of Optimizer class, containing the SSA 1 splitting-type method with constant momentum coefficient
class SSA2Nonconst(optimizer.Optimizer):
    # The constructor of the class
    def __init__(self, model, learning_rate = 1e-2, k = 2.0, alpha = 4.0, use_locking = False, name = 'SSA2Nonconst'):
        # Call the constructor of the 'Optimizer' superclass using the parameters 'use_locking' and 'name'
        super(SSA2Nonconst, self).__init__(use_locking, name)
        # Initialize the private Python variables of the current subclass
        self._lr = learning_rate
        self._alpha = alpha
        self._k = k
        self._model = model


        # Initialize the private 'Tensor' objects of the current subclass
        self._lr_t = None
        self._alpha_t = None
        self._k_t = None


    # We construct all the 'Tensor' objects before we apply the gradients
    # Private function
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name = 'learning_rate')
        self._alpha_t = ops.convert_to_tensor(self._alpha, name = 'alpha')
        self._k_t = ops.convert_to_tensor(self._k, name = 'k')

    # We create the slots for the variables. A 'Slot' is an additional variable associated with the variables to train
    # We allocate and manage these auxiliary variables
    # Private function
    def _create_slots(self, var_list):
        for v in var_list:
            # The accumulator variable is the accumulator obtained from the discrete-type velocity. It is denoted by 'p^{k+1}'
            self._zeros_slot(v, "old_accum", self._name)
            self._zeros_slot(v, "accum", self._name)
            self._zeros_slot(v, "curr_it", self._name)

    # The actual SSA 2 implementation for the general case when we have dense 'Tensor' objects
    # All of the operations are applied to 'Tensor' variables

    # 'apply_gradients', 'compute_gradients' and 'minimize' are public functions of 'Optimizer' class
    # Order of functions:
    # minimize(loss, global_step, var_list)
    # => grads_and_vars = compute_gradients(loss, var_list)
    # => grads_and_vars = list(zip(grads, var_list))
    # => grads = gradients.gradients(loss, var_refs)
    # var_list = (variables.trainable_variables() + ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

    # apply_gradients(grads_and_vars, global_step)
    # => for g, v in grads_and vars: p = _get_processor(v)
    # => _TensorProcessor(v), _DenseResourceVariableProcessor(v), _DenseResourceVariableProcessor(v), _RefVariableProcessor(v)
    # => for grad, var, processor in converted_grads_and_vars: processor.update_op(grad)
    # => update_op(self, optimizer, g)
    # => return update_op = optimizer._apply_dense(g, self._v)
    def _apply_dense(self, grad, var):
        # 1st step: we convert our 'Tensor' objects to have the type of the training variables
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        k_t = math_ops.cast(self._k_t, var.dtype.base_dtype)

        # 2nd step: we define the gradient accumulations, using the identifiers 'old_accum' and 'accum' from '_create_slots()'
        old_accum = self.get_slot(var, "old_accum")
        accum = self.get_slot(var, "accum")

        # 3rd step: define the current iteration needed for the momentum inertial sequence
        # It must be converted to the same type as the trainable variables
        # We have here the inertial sequences 'alpha_n' and 'beta_n'
        curr_it = self.get_slot(var, "curr_it")
        n = curr_it + 1
        mom = n / (n + alpha_t)
        new_mom = (n + 1.0) / (n + alpha_t + 1.0)

        # 4th step: we have the SSA2 constant momentum formula 'accum_t <- beta_{n+1} * beta_n^k * (1/beta_n - lr) * old_accum - lr^2 * beta_{n+1} beta_n^k * grad ', i.e. 'p^{k+1}' from Defazio
        old_accum_t = state_ops.assign(old_accum, accum)
        with ops.control_dependencies([old_accum_t]):
            accum_t = state_ops.assign(accum, new_mom * (mom ** k_t) * (1/mom - lr_t) * accum - (lr_t ** 2) * new_mom * (mom ** k_t) * grad , use_locking = False)


        # 5th step: variables updates by using 'var_update <- var + ( accum + old_accum * (1/beta_n - lr - 1) )', i.e. 'x^{k+1}' from Defazio
        # Here, 'accum_t' is 'p^{k+1}' because was already updated before
        # We use 'state_ops.add' instead of 'state_ops.sub'
        with ops.control_dependencies([old_accum, accum_t]):
            var_update = state_ops.assign_add(var, accum_t + old_accum_t * (1/mom - lr_t - 1))

        # 6th step: return the updates, i.e. we return the Graph 'Operation' that will group multiple 'Tensor' ops.
        # For more complex algorithms, the 'control_flow_ops.group' is used in the '_finish()' function, after '_apply_dense()'
        return control_flow_ops.group(*[var_update, old_accum_t, accum_t, n])


    # I did not implemented the algorithm for the case of 'Sparse Tensor' variables
    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


