''''
                                    DOCUMENTATION:
The so-called INDIAN algorithm appeared in the following paper:
https://arxiv.org/pdf/1905.12278.pdf

The Tensorflow implementation was done in the following GitHub repository:
https://github.com/camcastera/Indian-for-DeepLearning/tree/master/indian_for_tensorflow

The associated evolution equation is Hessian-driven damping system with constant damping. The INDIAN discretization
is in fact explicit forward Euler method.

Default values for the parameters (as in the original GitHub repository):
1.) 'learning_rate = 1e-2
2.) 'alpha = 0.5'
3.) 'beta = 0.1'
4.) 'gamma = 1.0'
5.) 'gamma_power = 0.5'
6.) 'init_velocity = 1.0'


'''


# Loading modules
from tensorflow.python.training import optimizer    # Here we have the 'Optimizer' class
from tensorflow.python.framework import ops         # From here we need the function that converts to 'Tensor' object
from tensorflow.python.ops import math_ops          # From here we need mathematical operations for 'Tensor' objects
from tensorflow.python.ops import state_ops         # From here we need 'Operations' on 'Tensor' objects
from tensorflow.python.ops import control_flow_ops  # From here we need the function 'group'
from tensorflow.train import get_or_create_global_step as curr_it    # Get the current iteration function
from tensorflow import cond                         # This is used in the definition of 'psi_k' at iteration no. 0
from tensorflow.math import equal                   # This is used in the definition of 'psi_k' at iteration no. 0




# The subclass of Optimizer class, containing the original INDIAN method
class Indian(optimizer.Optimizer):
    # The constructor of the class
    def __init__(self, model, learning_rate = 1e-2, alpha = 0.5, beta = 0.1, gamma_0 = 1.0, gamma_power = 0.5, init_velocity = 1.0, use_locking = False, name = 'Indian'):
        # Call the constructor of the 'Optimizer' superclass using the parameters 'use_locking' and 'name'
        super(Indian, self).__init__(use_locking, name)
        # Initialize the private Python variables of the current subclass
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta
        self._gamma_0 = gamma_0
        self._gamma_power = gamma_power
        self._init_velocity = init_velocity
        self._model = model


        # Initialize the private 'Tensor' objects of the current subclass
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None
        self._gamma_0_t = None
        self._gamma_power_t = None
        self._init_velocity_t = None


    # We construct all the 'Tensor' objects before we apply the gradients
    # Except the learning rate, all the coefficients are part of the momentum-type terms
    # Private function
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name = 'learning_rate')
        self._alpha_t = ops.convert_to_tensor(self._alpha, name = 'alpha')
        self._beta_t = ops.convert_to_tensor(self._beta, name = 'beta')
        self._gamma_0_t = ops.convert_to_tensor(self._gamma_0, name = 'gamma_0')
        self._gamma_power_t = ops.convert_to_tensor(self._gamma_power, name = 'gamma_power')
        self._init_velocity_t = ops.convert_to_tensor(self._init_velocity, name = 'init_velocity')


    # We create the slots for the variables. A 'Slot' is an additional variable associated with the variables to train
    # We allocate and manage these auxiliary variables
    # Private function
    def _create_slots(self, var_list):
        for v in var_list:
            # The accumulator variable is 'psi_{k+1}' from the original work of Castera et. al. - see relation (7) from page 7
            self._zeros_slot(v, "psi", self._name)


    # The actual INDIAN optimization algorithm implementation for the general case when we have dense 'Tensor' objects
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
        gamma_0_t = math_ops.cast(self._gamma_0_t, var.dtype.base_dtype)
        gamma_power_t = math_ops.cast(self._gamma_power_t, var.dtype.base_dtype)
        init_velocity_t = math_ops.cast(self._init_velocity_t, var.dtype.base_dtype)


        # 2nd step: we define the gradient accumulations, using the identifier 'psi' from '_create_slots()'
        # We also memorize the old accumulator, since we will update the 'psi' variable. Here, 'psi_k' is defined in Castera et. al.
        psi_k = self.get_slot(var, "psi")


        # 3rd step: define the current iteration needed for the momentum inertial sequence
        # It must be converted to the same type as the trainable variables
        n = math_ops.cast(curr_it(), var.dtype.base_dtype)

        # Here, we consider the adaptive learning rate, denoted by 'gamma_k' in the original article
        # The formula is given in the upper part of page 10
        adaptive_lr = lr_t * gamma_0_t / math_ops.pow(n+1 , gamma_power_t)

        # 4th step: initialize the old value of 'psi_k', depending on the current iteration (if it is equal to 0 or not)
        # If the number of iterations > 0, then 'psi_k' = psi (the additional 'Slot' variable)
        # the formula for iteration no. 0 is given in the upper part of page 20
        # A correction: we must have '(beta^2 - beta * initial_velocity) * grad' in order to match the original implementation
        psi_cond = cond(equal(n, 0), lambda: (1.0 - alpha_t * beta_t) * var - beta_t ** 2 * grad + beta_t * init_velocity_t * grad, lambda: psi_k)

        # 5th step: we have the INDIAN formula 'psi_k <- psi_k - adaptive_lr * ((alpha - 1/beta) * theta_k + 1/beta * psi_k)'
        # We update 'accum' by assigning the value 'momentum_t * accum + grad' to it. Furthermore, the new value is return in the 'Tensor' object 'accum_t'
        with ops.control_dependencies([psi_cond]):
            psi_k_plus_one = psi_k.assign(psi_cond - adaptive_lr * ((alpha_t - 1.0 / beta_t) * var + 1.0 / beta_t * psi_cond))

        # 6th step: variables updates by using 'theta_k <- theta_k - adaptive_lr * ( (alpha-1/beta) * theta_k  + 1/beta * psi_k + beta * grad)'
        # Here we use 'state_ops.assign_sub', so the sign of the coefficients is opposite to the one appearing in the 3rd line of relation (7)
        # Here, 'grad' is in fact 'v_k' from the algorithm (7)
        with ops.control_dependencies([psi_cond]):
            var_update = state_ops.assign_sub(var, adaptive_lr * ((alpha_t - 1.0 / beta_t) * var + 1.0 / beta_t * psi_cond + beta_t * grad))

        # 7th step: return the updates, i.e. we return the Graph 'Operation' that will group multiple 'Tensor' ops.
        # For more complex algorithms, the 'control_flow_ops.group' is used in the '_finish()' function, after '_apply_dense()'
        return control_flow_ops.group(*[var_update, psi_k_plus_one])


    # I did not implemented the algorithm for the case of 'Sparse Tensor' variables
    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


