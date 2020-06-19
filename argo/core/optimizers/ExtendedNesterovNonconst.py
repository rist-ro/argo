'''
                                    DOCUMENTATION:
Nesterov method with constant momentum factor is given in the work of Defazio:
https://arxiv.org/abs/1812.04634
See Table 1, page 3 - 'Modern Momentum' (here: beta is the momentum factor)

The Extended Nesterov method (without prox argument), is given in the following articles:
https://arxiv.org/pdf/1908.02574.pdf

Here we have 2 non-constant momentum-type sequences, 'alpha_n' and 'beta_n' that depend on 3 coefficients:
'alpha > 0', 'beta in R' and 'gamma >= 0'.
We impose, as in the continuous version, the following conditions: 'alpha > 3', 'beta in R' and 'gamma > 0'.

Some notes on default values and preliminary computational simulations:
1) I have put the default value of 'alpha' is 4, since 'alpha > 3'
2) For 'beta = 3' or similar high values, it does not work. I have put the default value to 'beta = 0.1'
3) A moderate ~ small value of 'gamma = 0.5' seems goo enough

'''



# Loading modules
from tensorflow.python.training import optimizer    # Here we have the 'Optimizer' class
from tensorflow.python.framework import ops         # From here we need the function that converts to 'Tensor' object
from tensorflow.python.ops import math_ops          # From here we need mathematical operations for 'Tensor' objects
from tensorflow.python.ops import state_ops         # From here we need 'Operations' on 'Tensor' objects
from tensorflow.python.ops import control_flow_ops  # From here we need the function 'group'



# The subclass of Optimizer class, containing Nesterov method with constant momentum coefficients, namely 'alpha' and 'beta'
class ExtendedNesterovNonconst(optimizer.Optimizer):
    # The constructor of the class
    def __init__(self, model, learning_rate = 1e-2, alpha = 4, beta = 0.1, gamma = 0.5, use_locking = False, name = 'ExtendedNesterovNonconst'):
        # Call the constructor of the 'Optimizer' superclass using the parameters 'use_locking' and 'name'
        super(ExtendedNesterovNonconst, self).__init__(use_locking, name)
        # Initialize the private Python variables of the current subclass
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._model = model

        # Initialize the private 'Tensor' objects of the current subclass
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None
        self._gamma_t = None


    # We construct all the 'Tensor' objects before we apply the gradients
    # Except the learning rate, all the coefficients are part of the momentum-type terms
    # Private function
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name = 'learning_rate')
        self._alpha_t = ops.convert_to_tensor(self._alpha, name = 'alpha')
        self._beta_t = ops.convert_to_tensor(self._beta, name = 'beta')
        self._gamma_t = ops.convert_to_tensor(self._gamma, name = 'gamma')


    # We create the slots for the variables. A 'Slot' is an additional variable associated with the variables to train
    # We allocate and manage these auxiliary variables
    # Private function
    def _create_slots(self, var_list):
        for v in var_list:
            # The accumulator variable is 'p^{k+1}' in the work of Defazio
            self._zeros_slot(v, "old_accum", self._name)
            self._zeros_slot(v, "accum", self._name)
            self._zeros_slot(v, "curr_it", self._name)

    # The actual Extended Nesterov implementation for the general case when we have dense 'Tensor' objects
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
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)


        # 2nd step: we define the gradient accumulations, using the identifier 'accum' from '_create_slots()'
        # We also memorize the old accumulator, since we will update the 'accum' variable. Here, 'old_accum' is 'p^{k+1}'
        old_accum = self.get_slot(var, "old_accum")
        accum = self.get_slot(var, "accum")


        # 3rd step: define the current iteration needed for the momentum inertial sequence
        # It must be converted to the same type as the trainable variables
        # We have here the inertial sequences 'alpha_n' and 'beta_n'
        curr_it = self.get_slot(var, "curr_it")
        n = curr_it + 1

        alpha_iteration = n / (n + alpha_t)
        beta_iteration = (n * gamma_t + beta_t) / (n + alpha_t)
        beta_iteration_plus_1 = ((n+1) * gamma_t + beta_t) / (n + alpha_t + 1)


        # 4th step: we have the Extended Nesterov formula 'accum_t <- accum_t * alpha_t + grad', i.e. 'p^{k+1}' from Defazio
        # We update 'accum' by assigning the value 'momentum_t * accum + grad' to it. Furthermore, the new value is return in the 'Tensor' object 'accum_t'
        old_accum_t = state_ops.assign(old_accum, accum)
        with ops.control_dependencies([old_accum_t]):
            accum_t = state_ops.assign(accum, alpha_iteration * accum + grad, use_locking = False)

        # 5th step: variables updates by using 'var_update <- var - ( lr_t * grad + lr_t * beta_t * accum_t + (alpha_t-beta_t) * old_accum )', i.e. 'x^{k+1}' from Defazio
        # Here, 'accum_t' is 'p^{k+1}' because was already updated before
        with ops.control_dependencies([old_accum, accum_t]):
            var_update = state_ops.assign_sub(var, lr_t * grad + lr_t * beta_iteration_plus_1 * accum_t + (alpha_iteration - beta_iteration) * old_accum_t)

        # 6th step: return the updates, i.e. we return the Graph 'Operation' that will group multiple 'Tensor' ops.
        # For more complex algorithms, the 'control_flow_ops.group' is used in the '_finish()' function, after '_apply_dense()'
        return control_flow_ops.group(*[var_update, old_accum_t, accum_t, n])


    # I did not implemented the algorithm for the case of 'Sparse Tensor' variables
    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


