from .AbstractModule import AbstractModule
from ..utils.argo_utils import update_conf_with_defaults


class ArgoAbstractNetwork(AbstractModule):
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

    # @abstractmethod
    def create_id(self):
        return ""