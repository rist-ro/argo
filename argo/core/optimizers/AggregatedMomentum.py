''''
                                    DOCUMENTATION:
The implemented method is called 'Aggregated Momentum'. See the following arXiv link:
https://arxiv.org/pdf/1804.00325.pdf

Furthermore, the method was already implemented in (copyright?):
https://github.com/AtheMathmo/AggMo



1.) The default value of 'a' is set to 0.1, such that the 'betas' vector contain positive values up to 1.
E.g. : learning = 0.7 and K = 10 (lower learning rate for increased value of the momentum terms)

2.) Finally, I did not implement the 'apply_shared' function (TODO).

3.) I have made some modifications to the code.
I have implemented in a basic manner. I did not used the @classmethod decorator,
so if my method is slower, just implement the original method.
(in general, the decorator @classmethod is used for the inherent parameters of the class that have a well defined structure)

4.) If I am not mistaking, since 'var_update' is defined through 'state_ops.assign_sub',
then at 'm_t' we need '-betas[i]', since the gradient has positive sign and the 'betas' must have the reversed sign.
I did not implemented this valid defintion (as it was introduced in the article),
since when 'betas' and grad have the same sign, it works better!
(in fact, in this way Aggretated Momentum was implemented in the GitHub link, as explained above)

'''


# Loading modules
from tensorflow.python.training import optimizer    # Here we have the 'Optimizer' class
from tensorflow.python.framework import ops         # From here we need the function that converts to 'Tensor' object
from tensorflow.python.ops import math_ops          # From here we need mathematical operations for 'Tensor' objects
from tensorflow.python.ops import state_ops         # From here we need 'Operations' on 'Tensor' objects
from tensorflow.python.ops import control_flow_ops  # From here we need the function 'group'



# The subclass of Optimizer class, containing the Aggregated momentum method
class AggregatedMomentum(optimizer.Optimizer):
    # The constructor of the class
    def __init__(self, model, learning_rate = 1e-3, a = 0.1, K = 3, use_locking = False, name = 'AggregatedMomentum'):
        # Call the constructor of the 'Optimizer' superclass using the parameters 'use_locking' and 'name'
        super(AggregatedMomentum, self).__init__(use_locking, name)
        # Initialize the private Python variables of the current subclass
        self._lr = learning_rate
        self._a = a
        self._K = K
        self._betas = [1.0 - a ** i for i in range(K)]
        self._model = model


        # Initialize the private 'Tensor' objects of the current subclass
        self._lr_t = None
        self._a_t = None
        self._K_t = None
        self._betas_t = [None for i in range(K)]


    # We construct all the 'Tensor' objects before we apply the gradients
    # Private function
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name = 'learning_rate')
        self._a_t = ops.convert_to_tensor(self._a, name = 'a')
        self._K_t = ops.convert_to_tensor(self._K, name = 'K')
        self._betas_t = [ops.convert_to_tensor(self._betas[i], name = 'betas_{}'.format(i)) for i in range(self._K)]


    # We create the slots for the variables. A 'Slot' is an additional variable associated with the variables to train
    # We allocate and manage these auxiliary variables
    # Private function
    def _create_slots(self, var_list):
        for v in var_list:
            # The K momentum variables are stored as 'Slots'
            for i in range(self._K):
                self._zeros_slot(v, "momentum_{}".format(i), self._name)


    # The actual Aggregated momentum implementation for the general case when we have dense 'Tensor' objects
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
        K_t = math_ops.cast(self._a_t, var.dtype.base_dtype)
        betas_t = [math_ops.cast(self._betas_t[i], var.dtype.base_dtype) for i in range(self._K)]

        #2nd step: we consider here the adaptive learning rate
        adaptive_lr = lr_t / K_t

        #3rd step: we define a list in which we append the new K momentum variables, along with their sum
        # We do not store them nor as 'Slots', neither as additional 'Non-Slot' variables
        # They are used at each step and then we can discard them
        momentum_list = []
        summed_momentum = 0.0
        for i in range(self._K):
            m = self.get_slot(var, "momentum_{}".format(i))
            m_t = state_ops.assign(m, - betas_t[i] * m - grad)
            summed_momentum += m_t
            momentum_list.append(m_t)

        # 4th step: variables updates by using 'var_update <- var + ( adaptive_lr * summed_momentum )'
        # Here, 'accum_t' is 'p^{k+1}' because was already updated before
        var_update = state_ops.assign_add(var, adaptive_lr * summed_momentum, use_locking=self._use_locking)

        # 5th step: return the updates, i.e. we return the Graph 'Operation' that will group multiple 'Tensor' ops.
        # For more complex algorithms, the 'control_flow_ops.group' is used in the '_finish()' function, after '_apply_dense()'
        # Here we have put * in front of the 'momentum_list' which is of length K, since in '_create_slots()', we have constructed K elements
        return control_flow_ops.group(*[var_update, *momentum_list])


    # I did not implemented the algorithm for the case of 'Sparse Tensor' variables
    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")



