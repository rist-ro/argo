
from .AbstractStochasticLayer import AbstractStochasticLayer

# from .utils.argo_utils import NUMTOL, get_ac_collection_name, eval_method_from_tuple, load_sonnet_module


class AbstractLogistic(AbstractStochasticLayer):
    
    def __init__(self, module_tuple = ("Linear", {}),
                 output_size=-1,
                 output_shape=-1,
                 minimal_covariance=0,
                 covariance_parameterization="softplus",
                 scalar_covariance = False,
                 initializers = {},
                 regularizers = {},
                 contractive_regularizer = None,
                 name='abstract_logistic'):
        
        super().__init__(module_tuple = module_tuple,
                         output_size = output_size,
                         output_shape = output_shape,
                         minimal_covariance = minimal_covariance,
                         covariance_parameterization = covariance_parameterization,
                         scalar_covariance = scalar_covariance,
                         initializers = initializers,
                         regularizers = regularizers,
                         contractive_regularizer = contractive_regularizer)
    
    @property
    def _contractive_regularizer_filename(self):
        return ".LogisticRegularizers"
    
    
