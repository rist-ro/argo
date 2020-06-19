import tensorflow as tf

from .utils.argo_utils import eval_method_from_tuple

import importlib

import pdb

class CostFunctions():

    @staticmethod
    def instantiate_cost_function(cost_function_tuple, module_path = ""):

        cost_function_name = cost_function_tuple[0]
        cost_function_kwargs = cost_function_tuple[1]

        try:

            # first try to load from core
            try:
                cost_function_module = importlib.import_module("core." + cost_function_name, '.'.join(__name__.split('.')[:-1]))
            # it if fails, try to laod from module_path.core
            except ImportError:
                try:
                    cost_function_module = importlib.import_module("core.cost." + cost_function_name,
                                                                   '.'.join(__name__.split('.')[:-1]))
                except ImportError:
                    cost_function_module = importlib.import_module(module_path + ".core." + cost_function_name,
                                                                   '.'.join(__name__.split('.')[:-1]))

            cost_function = eval_method_from_tuple(cost_function_module, cost_function_tuple)

        except Exception as e:
            raise Exception("problem with module: %s, kwargs: %s, exception %s" % (cost_function_name, cost_function_kwargs, e)) from e

        return cost_function

    # @staticmethod
    # def create_id(cost_function_tuple):
    #     cost_function = CostFunctions.instantiate_cost_function(cost_function_tuple)
    #     return cost_function.create_id(cost_function_tuple[1])
