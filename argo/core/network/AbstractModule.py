import sonnet as snt
import tensorflow as tf


class AbstractModule(snt.AbstractModule):
    """
    Sonnet Abstract Module
    """

    def get_all_variables(self, collection=tf.GraphKeys.GLOBAL_VARIABLES):
        """ overwrites the default collection to filter the variables,
        the default for Sonnet module is collection=tf.GraphKeys.TRAINABLE_VARIABLES
        that excludes moving mean and standard deviations for example.

        Returns:
            variables used when the module is connected

        see more in `AbstractModule.get_all_variables` documentation
        """
        return super().get_all_variables(collection=collection)

    def get_variables(self, collection=tf.GraphKeys.GLOBAL_VARIABLES):
        """ overwrites the default collection to filter the variables,
        the default for Sonnet module is collection=tf.GraphKeys.TRAINABLE_VARIABLES
        that excludes moving mean and standard deviations for example.

        Returns:
            variables defined in the scope of the Module

        see more in `AbstractModule.get_variables` documentation
        """
        return super().get_variables(collection=collection)

    def init_saver(self, variables=None):
        """This set the saver for a network. It has to be invoked only when the Module has been connected,
        otherwise it will raise an exception

        Args:
            variables (list): A list of variables to save and restore, by default all the variables of the module
            (instantiated till this precise moment) will be tracked

        Raises:
          NotConnectedError: If the module is not connected to the Graph.
        """
        self._ensure_is_connected()

        if not variables:
            variables = self.get_variables()

        self._saver = tf.train.Saver(variables)

    def save(self, sess, chkptfile, **kwargs):
        if self._saver is None:
            # raise Exception("saver must be initialized before attempt to save")
            self.init_saver()

        self._saver.save(sess, chkptfile, **kwargs)

    def restore(self, sess, chkptfile, **kwargs):
        if self._saver is None:
            # import pdb;pdb.set_trace()
            # raise Exception("saver must be initialized before attempt to restore")
            self.init_saver()

        self._saver.restore(sess, chkptfile, **kwargs)