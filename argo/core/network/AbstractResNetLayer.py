import tensorflow as tf
import sonnet as snt
from ..keras_models.keras_utils import get_renorm_clipping

class AbstractResNetLayer(snt.AbstractModule):
    """

    """

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 activation,
                 is_training,
                 name='res',
                 prob_drop=0.1,
                 bn_momentum=0.99,
                 bn_renormalization=True,
                 **extra_params):
        """

        Args:
            num_outputs (type): number of outputs of the module.
            name (type): module name.
            activation (type): activation used for the internal layers.
            **extra_params (type): alls the additional keyword arguments will be passed to the snt.Conv2D layers. (initializers, regularizers)

        """
        super().__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._prob_drop = prob_drop
        self._activation = activation
        self._extra_params = extra_params

        self._bn_renormalization = bn_renormalization
        # instantiate all the convolutional layers

        self._bn_momentum = bn_momentum

        self._renorm_clipping = None
        if self._bn_renormalization:
            self._renorm_clipping = get_renorm_clipping()

        self._is_training = is_training

    def _dropout(self, net, training):
        net = tf.layers.dropout(net, self._prob_drop, training=training)
        return net
