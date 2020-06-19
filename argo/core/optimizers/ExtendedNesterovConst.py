'''
                                    DOCUMENTATION:
Nesterov method with constant momentum factor is given in the work of Defazio:
https://arxiv.org/abs/1812.04634
See Table 1, page 3 - 'Modern Momentum' (here: beta is the momentum factor)

The Extended Nesterov method (without prox argument), is given in the following articles:
https://arxiv.org/abs/2002.07154
https://link.springer.com/article/10.1007/s11075-019-00765-z


Here, the sequences 'alpha_n' and 'beta_n' are constant momentum-type factors. Can take negative values as well.


Some notes:

1.) From the theory, if we suppose the well-conditioning of the problem, i.e. the Lipschitz constant L=1, then we observe
that for 'alpha = 0.5' we have divergence. So, indeed 'alpha' is in (-0.5, 0.5).
Furthermore, since we want a small restriction on the stepsize, we need a small value for 'beta', e.g. 0.001
(quite interestingly, it works also for 300 epochs - at some weights and biases initializations, we have exploding gradients)

2.) It works only for very small values of negative 'alpha', e.g. 'alpha = -0.1' (but works even for 300 epochs)

3.) Default values (obtained after preliminary experiments):
The default value for positive 'alpha' is 0.2
The default value for negative 'alpha' is -0.1
The default value for beta is '0.01'

4.) For 'alpha = 0.2' and 'beta = 0.01', we have obtained the training accuracy 100.00000000000028% (rounding errors ??)
'''



# Loading modules
from tensorflow.python.training import optimizer    # Here we have the 'Optimizer' class
from tensorflow.python.framework import ops         # From here we need the function that converts to 'Tensor' object
from tensorflow.python.ops import math_ops          # From here we need mathematical operations for 'Tensor' objects
from tensorflow.python.ops import state_ops         # From here we need 'Operations' on 'Tensor' objects
from tensorflow.python.ops import control_flow_ops  # From here we need the function 'group'



# The subclass of Optimizer class, containing Nesterov method with constant momentum coefficients, namely 'alpha' and 'beta'
class ExtendedNesterovConst(optimizer.Optimizer):
    # The constructor of the class
    def __init__(self, model, learning_rate = 1e-2, alpha = 0.2, beta = 0.01, use_locking = False, name = 'ExtendedNesterovConst'):
        # Call the constructor of the 'Optimizer' superclass using the parameters 'use_locking' and 'name'
        super(ExtendedNesterovConst, self).__init__(use_locking, name)
        # Initialize the private Python variables of the current subclass
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta
        self._model = model

        # Initialize the private 'Tensor' objects of the current subclass
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    # We construct all the 'Tensor' objects before we apply the gradients
    # Private function
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name = 'learning_rate')
        self._alpha_t = ops.convert_to_tensor(self._alpha, name = 'alpha')
        self._beta_t = ops.convert_to_tensor(self._beta, name = 'beta')


    # We create the slots for the variables. A 'Slot' is an additional variable associated with the variables to train
    # We allocate and manage these auxiliary variables
    # Private function
    def _create_slots(self, var_list):
        for v in var_list:
            # The accumulator variable is 'p^{k+1}' in the work of Defazio
            self._zeros_slot(v, "old_accum", self._name)
            self._zeros_slot(v, "accum", self._name)

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

        # 2nd step: we define the gradient accumulations, using the identifier 'accum' from '_create_slots()'
        # We also memorize the old accumulator, since we will update the 'accum' variable. Here, 'old_accum' is 'p^{k+1}'
        old_accum = self.get_slot(var, "old_accum")
        accum = self.get_slot(var, "accum")


        # 3rd step: we have the Extended Nesterov formula 'accum_t <- accum_t * alpha_t + grad', i.e. 'p^{k+1}' from Defazio
        # We update 'accum' by assigning the value 'momentum_t * accum + grad' to it. Furthermore, the new value is return in the 'Tensor' object 'accum_t'
        old_accum_t = state_ops.assign(old_accum, accum)
        with ops.control_dependencies([old_accum_t]):
            accum_t = state_ops.assign(accum, alpha_t * accum + grad, use_locking = False)

        # 4th step: variables updates by using 'var_update <- var - ( lr_t * grad + lr_t * beta_t * accum_t + (alpha_t-beta_t) * old_accum )', i.e. 'x^{k+1}' from Defazio
        # Here, 'accum_t' is 'p^{k+1}' because was already updated before
        with ops.control_dependencies([old_accum, accum_t]):
            var_update = state_ops.assign_sub(var, lr_t * grad + lr_t * beta_t * accum_t + (alpha_t - beta_t) * old_accum_t)

        # 5th step: return the updates, i.e. we return the Graph 'Operation' that will group multiple 'Tensor' ops.
        # For more complex algorithms, the 'control_flow_ops.group' is used in the '_finish()' function, after '_apply_dense()'
        return control_flow_ops.group(*[var_update, old_accum_t, accum_t])


    # I did not implemented the algorithm for the case of 'Sparse Tensor' variables
    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


