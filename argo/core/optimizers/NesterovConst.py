'''
                                    DOCUMENTATION:
Nesterov method with constant momentum factor is given in the work of Defazio:
https://arxiv.org/abs/1812.04634
See Table 1, page 3 - 'Modern Momentum' (here: beta is the momentum factor)

The momentum coefficient is set between 0.5 and 0.9. See the work of Ruder:
https://arxiv.org/abs/1609.04747

'''



# Loading modules
from tensorflow.python.training import optimizer    # Here we have the 'Optimizer' class
from tensorflow.python.framework import ops         # From here we need the function that converts to 'Tensor' object
from tensorflow.python.ops import math_ops          # From here we need mathematical operations for 'Tensor' objects
from tensorflow.python.ops import state_ops         # From here we need 'Operations' on 'Tensor' objects
from tensorflow.python.ops import control_flow_ops  # From here we need the function 'group'



# The subclass of Optimizer class, containing Nesterov method with constant momentum coefficient
class NesterovConst(optimizer.Optimizer):
    # The constructor of the class
    def __init__(self, model, learning_rate = 1e-2, momentum = 0.5, use_locking = False, name = 'NesterovConst'):
        # Call the constructor of the 'Optimizer' superclass using the parameters 'use_locking' and 'name'
        super(NesterovConst, self).__init__(use_locking, name)
        # Initialize the private Python variables of the current subclass
        self._lr = learning_rate
        self._momentum = momentum
        self._model = model

        # Initialize the private 'Tensor' objects of the current subclass
        self._lr_t = None
        self._momentum_t = None


    # We construct all the 'Tensor' objects before we apply the gradients
    # Private function
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name = 'learning_rate')
        self._momentum_t = ops.convert_to_tensor(self._momentum, name = 'momentum')


    # We create the slots for the variables. A 'Slot' is an additional variable associated with the variables to train
    # We allocate and manage these auxiliary variables
    # Private function
    def _create_slots(self, var_list):
        for v in var_list:
            # The accumulator variable is 'p^{k+1}' in the work of Defazio
            self._zeros_slot(v, "accum", self._name)

    # The actual Nesterov implementation for the general case when we have dense 'Tensor' objects
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
        momentum_t = math_ops.cast(self._momentum_t, var.dtype.base_dtype)

        # 2nd step: we define the gradient accumulations, using the identifier 'accum' from '_create_slots()'
        accum = self.get_slot(var, "accum")

        # 3rd step: we have the Nesterov formula 'accum_t <- accum_t * momentum_t + grad', i.e. 'p^{k+1}' from Defazio
        # We update 'accum' by assigning the value 'momentum_t * accum + grad' to it. Furthermore, the new value is return in the 'Tensor' object 'accum_t'
        accum_t = state_ops.assign(accum, momentum_t * accum + grad, use_locking = False)

        # 4th step: variables updates by using 'var_update <- var - ( lr_t * grad + lr_t * momentum_t * accum_t )', i.e. 'x^{k+1}' from Defazio
        # Here, 'accum_t' is 'p^{k+1}' because was already updated before
        var_update = state_ops.assign_sub(var, lr_t * grad + lr_t * momentum_t * accum_t)

        # 5th step: return the updates, i.e. we return the Graph 'Operation' that will group multiple 'Tensor' ops.
        # For more complex algorithms, the 'control_flow_ops.group' is used in the '_finish()' function, after '_apply_dense()'
        return control_flow_ops.group(*[var_update, accum_t])


    # I did not implemented the algorithm for the case of 'Sparse Tensor' variables
    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


