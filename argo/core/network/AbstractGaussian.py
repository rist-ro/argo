from .AbstractStochasticLayer import AbstractStochasticLayer

class AbstractGaussian(AbstractStochasticLayer):

    def __init__(self, module_tuple=("Linear", {}),
                 output_size=None,
                 output_shape=None,
                 minimal_covariance=0,
                 covariance_parameterization="softplus",
                 scalar_covariance=False,
                 initializers={},
                 regularizers={},
                 contractive_regularizer=None,
                 name='abstract_gaussian'):

        super().__init__(module_tuple=module_tuple,
                         output_size=output_size,
                         output_shape=output_shape,
                         minimal_covariance=minimal_covariance,
                         covariance_parameterization=covariance_parameterization,
                         scalar_covariance=scalar_covariance,
                         initializers=initializers,
                         regularizers=regularizers,
                         contractive_regularizer=contractive_regularizer)

    @property
    def _contractive_regularizer_filename(self):
        return ".GaussianRegularizers"
    
