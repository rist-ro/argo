from .AbstractModule import AbstractModule


class AbstractGaussianSimple(AbstractModule):

    def __init__(self,
                 output_size,
                 minimal_covariance=1e-4,
                 initializers={},
                 regularizers={},
                 custom_getter={},
                 name='abstract_gaussian_linear'):

        super().__init__(name=name)

        self._output_size = output_size

        self._extra_kwargs = {"initializers": initializers,
                              "regularizers": regularizers,
                              "custom_getter" : custom_getter}

        self._minimal_covariance = minimal_covariance

