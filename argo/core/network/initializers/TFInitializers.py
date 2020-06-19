import importlib

import tensorflow as tf

from ...utils.argo_utils import eval_method_from_tuple

method_name_short = {
    # initializers
    "glorot_normal_initializer":    'gn',
    "glorot_uniform_initializer":   'gu',
    "xavier_initializer":           'x',
    "truncated_normal_initializer": 't',
    "variance_scaling_initializer": 'v',
    "VarianceScaling":              'v',
    "constant_initializer":         'c',
    "constant":                     'c',
    "random_normal":                'n',
}


class TFInitializers():

    @staticmethod
    def instantiate_initializer(initializer_tuple):

        initializer_name = initializer_tuple[0]
        initializer_kwargs = initializer_tuple[1]

        # I want to copy because I want to modify it and I don't want to accidentally modify all the references around
        # in python references to a particular entry of a dictionary can be passed around and I might overwrite different task_opts
        initializer_kwargs = initializer_kwargs.copy()

        try:
            # try to get the module from tf.train
            initializer = eval_method_from_tuple(tf, (initializer_name, initializer_kwargs))
        except AttributeError as e:

            try:
                # first try to load from argo.core.initializers
                initializer_module = importlib.import_module("." + initializer_name, '.'.join(__name__.split('.')[:-1]))

                initializer = eval_method_from_tuple(initializer_module, (initializer_name, initializer_kwargs))

            except ImportError:
                # second try to load from core.initializers
                initializer_module = importlib.import_module("core.network.initializers." + initializer_name,
                                                             '.'.join(__name__.split('.')[:-1]))

                initializer = eval_method_from_tuple(initializer_module, (initializer_name, initializer_kwargs))

            except Exception as e:
                raise Exception("problem loading initializer: %s, kwargs: %s, exception: %s" % (
                    initializer_name, initializer_kwargs, e)) from e

        return initializer

    @staticmethod
    def create_id(initializer_tuple):
        initializer_name, initializer_kwargs = initializer_tuple

        initializer_name = initializer_name.split(".")[-1]

        _id = ''

        if initializer_name == 'PlusMinusOneConstantInitializer':
            _id += 'PMOCI'
            _id += '_w{}'.format(initializer_kwargs["w"])
            _id += '_b{}'.format(initializer_kwargs["b"])


        elif initializer_name in method_name_short:

            _id += method_name_short[initializer_name]

            if "xavier_initializer" in initializer_name:
                pass

            elif "variance_scaling_initializer" in initializer_name or "VarianceScaling" in initializer_name:
                _id += "S"+str(initializer_kwargs['scale'])+"M"+initializer_kwargs['mode'][-3:]+"D"+initializer_kwargs['distribution'][:3]

            elif "glorot_normal_initializer" in initializer_name or "glorot_uniform_initializer" in initializer_name:
                pass

            elif "truncated_normal_initializer" in initializer_name or "random_normal" in initializer_name:
                _id += str(initializer_kwargs['stddev'])

            elif "constant_initializer" in initializer_name or "constant" in initializer_name:
                _id += str(initializer_kwargs['value'])
            else:
                raise Exception(
                    "No such initializer defined in tf or argo.core or your module! {}".format(initializer_name))

        else:
            raise Exception(
                "No such initializer defined in tf or argo.core or your module! {}".format(initializer_name))

        return _id
