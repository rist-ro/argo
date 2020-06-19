import tensorflow as tf

from .utils.argo_utils import eval_method_from_tuple

import importlib

import pdb

class Regularizers():

    @staticmethod
    def instantiate_regularizer(regularizer_tuple, module_path = ""):

        regularizer_name = regularizer_tuple[0]
        regularizer_kwargs = regularizer_tuple[1]

        try:
            # first try to load from core
            #try:
            regularizer_module = importlib.import_module(".regularizers." + regularizer_name, '.'.join(__name__.split('.')[:-3]))
            # it if fails, try to laod from module_path.core
            #except ImportError:
            #    regularizer_module = importlib.import_module(module_path + ".core." + regularizer_name, '.'.join(__name__.split('.')[:-1]))

            custom_regularizer, _, _= eval_method_from_tuple(regularizer_module, regularizer_tuple)

        except Exception as e:
            raise Exception("problem with module: %s, kwargs: %s, exception %s" % (regularizer_name, regularizer_kwargs, e)) from e

        return custom_regularizer


# generic regularizers: add implementation here, plus the create_id
# see how Regularizers in gan or vae


