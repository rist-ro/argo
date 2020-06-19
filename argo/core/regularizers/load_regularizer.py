import tensorflow as tf
from ..utils.argo_utils import eval_method_from_tuple

import importlib


def load_regularizer(regularizer_tuple): #, module_path = ""):

    regularizer_name = regularizer_tuple[0]
    regularizer_kwargs = regularizer_tuple[1]

    try:
        # first try to load from here
        try:
            regularizer_module = importlib.import_module("."+regularizer_name, '.'.join(__name__.split('.')[:-1]))
        # it if fails, try to load from tf
        except ImportError:
            # regularizer_module = importlib.import_module(module_path + ".core." + regularizer_name, '.'.join(__name__.split('.')[:-1]))
            regularizer_module = tf

        regularizer = eval_method_from_tuple(regularizer_module, regularizer_tuple)

    except Exception as e:
        raise Exception("problem with module: %s, kwargs: %s, exception %s" % (regularizer_name, regularizer_kwargs, e)) from e

    return regularizer


