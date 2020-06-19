import tensorflow as tf

from ..utils.argo_utils import update_conf_with_defaults


class ArgoAbstractKerasNetwork(tf.keras.Model):
    """
    Abstract class for managing a network in Argo
    """

    default_params = {}

    def __init__(self, opts, name, seed=None):
        super().__init__(name=name)

        self._opts_original = opts
        self._opts = update_conf_with_defaults(opts, self.default_params)

        self._seed = seed

        self._name = name
        # self._seed = seed
        self._saver = None

    # abstractmethod
    def create_id(self):
        return ""

    def get_variables(self):
        """ overwrites the default collection to filter the variables,
        the default for Sonnet module is collection=tf.GraphKeys.TRAINABLE_VARIABLES
        that excludes moving mean and standard deviations for example.

        Returns:
            variables defined in the scope of the Module

        see more in `AbstractModule.get_variables` documentation
        """

        return self.weights

    def get_all_variables(self):
        """ overwrites the default collection to filter the variables,
        the default for Sonnet module is collection=tf.GraphKeys.TRAINABLE_VARIABLES
        that excludes moving mean and standard deviations for example.

        Returns:
            variables defined in the scope of the Module

        see more in `AbstractModule.get_variables` documentation
        """

        return self.weights

    def init_saver(self, variables=None):
        """This set the saver for a network. It has to be invoked only when the Model has been connected

        Args:
            variables (list): A list of variables to save and restore, by default all the variables of the module
            (instantiated till this precise moment) will be tracked

        """

        if not variables:
            variables = self.get_variables()

        self._saver = tf.train.Saver(variables)

    def save_argo(self, sess, chkptfile, **kwargs):
        if self._saver is None:
            self.init_saver()

        self._saver.save(sess, chkptfile, **kwargs)

    def restore(self, sess, chkptfile, **kwargs):
        if self._saver is None:
            self.init_saver()

        self._saver.restore(sess, chkptfile, **kwargs)