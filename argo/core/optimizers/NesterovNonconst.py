'''
                                    DOCUMENTATION:
Nesterov method with non-constant momentum factor is the original one. It can be found in the original work of Y. Nesterov:
http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf

Nesterov method with non-constant momentum factor can be given similar to the one from Defazio:
https://arxiv.org/abs/1812.04634

Here we do not give the momentum factor as a parameter. We consider the construction given in the work of Chambolle and Dossal:
https://hal.inria.fr/hal-01060130v3/document
They have defined 'beta_n = (t_n - 1) / t_{n+1}', where 't_n = (n + alpha - 1) / alpha' => 'beta_n = (n - 1) / (n + alpha)' for n >= 1
Since the 'global_step' starts from 0, we take n >=0, and so we take 'beta_n = n / (n + alpha + 1)', where 'alpha' = given parameter

We mention that in the 'main' file we must initialize the global step as a 'Variable'

'''



# Loading modules
from tensorflow.python.training import optimizer    # Here we have the 'Optimizer' class
from tensorflow.python.framework import ops         # From here we need the function that converts to 'Tensor' object
from tensorflow.python.ops import math_ops          # From here we need mathematical operations for 'Tensor' objects
from tensorflow.python.ops import state_ops         # From here we need 'Operations' on 'Tensor' objects
from tensorflow.python.ops import control_flow_ops  # From here we need the function 'group'
from tensorflow.python.eager import context         # From here we need the option to verify if we are executing in 'Eager' mode



# The subclass of Optimizer class, containing Nesterov method with constant momentum coefficient
class NesterovNonconst(optimizer.Optimizer):
    # The constructor of the class
    def __init__(self, model, learning_rate = 1e-2, alpha = 2.0, use_locking = False, name = 'NesterovNonconst'):
        # Call the constructor of the 'Optimizer' superclass using the parameters 'use_locking' and 'name'
        super(NesterovNonconst, self).__init__(use_locking, name)
        # Initialize the private Python variables of the current subclass
        self._lr = learning_rate
        self._alpha = float(alpha)
        self._model = model


        # Initialize the private 'Tensor' objects of the current subclass
        self._lr_t = None
        self._alpha_t = None



    # Order of functions:
    # apply_gradients(grads_and_vars, global_step)
    # => self._create_slots(var_list)
    # and self._prepare()

    # We construct all the 'Tensor' objects before we apply the gradients
    # Private function
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name = 'learning_rate')
        self._alpha_t = ops.convert_to_tensor(self._alpha, name = 'alpha')



    # We create the slots for the variables. A 'Slot' is an additional variable associated with the variables to train
    # We allocate and manage these auxiliary variables
    # Private function
    def _create_slots(self, var_list):
        # Create 'Slot' variables
        for v in var_list:
            # The accumulator variable is 'p^{k+1}' in the work of Defazio
            self._zeros_slot(v, "accum", self._name)
            self._zeros_slot(v, "momentum", self._name)
            self._zeros_slot(v, "curr_it", self._name)

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
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

        # 2nd step: we define the gradient accumulations, using the identifier 'accum' from '_create_slots()'
        accum = self.get_slot(var, "accum")

        # Optional 'get_slot' for momentum. Then we can use 'momentum = momentum_t' and then use only 'accum_t' and 'momentum' at updates
        # momentum = self.get_slot(var, "momentum")

        # 3rd step: define the current iteration needed for the momentum inertial sequence
        # It must be converted to the same type as the trainable variables
        curr_it = self.get_slot(var, "curr_it")
        n = curr_it + 1


        momentum_t = n / ( n  + alpha_t + 1.0 )
        momentum_t_1 = (n+1) / ( n + alpha_t + 2.0 )

        # 4th step: we have the Nesterov formula 'accum_t <- accum_t * momentum_t + grad', i.e. 'p^{k+1}' from Defazio
        # We update 'accum' by assigning the value 'momentum_t * accum + grad' to it. Furthermore, the new value is return in the 'Tensor' object 'accum_t'
        accum_t = state_ops.assign(accum, momentum_t * accum + grad, use_locking = False)

        # 5th step: variables updates by using 'var_update <- var - ( lr_t * grad + lr_t * momentum_{t+1} * accum_t )', i.e. 'x^{k+1}' from Defazio
        # Here, 'accum_t' is 'p^{k+1}' because was already updated before
        var_update = state_ops.assign_sub(var, lr_t * grad + lr_t * momentum_t_1 * accum_t)

        # 6th step: return the updates, i.e. we return the Graph 'Operation' that will group multiple 'Tensor' ops.
        # For more complex algorithms, the 'control_flow_ops.group' is used in the '_finish()' function, after '_apply_dense()'
        return control_flow_ops.group(*[var_update, accum_t, momentum_t, n])


    # I did not implemented the algorithm for the case of 'Sparse Tensor' variables
    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
