import importlib
# from argo.core.utils.argo_utils import eval_method_from_tuple


def get_preproc_module(preproc_tuple, module_path=""):
    preproc_name = preproc_tuple[0]
    preproc_kwargs = preproc_tuple[1]

    try:
        # first try to load from here
        try:
            py_module = importlib.import_module("." + preproc_name, '.'.join(__name__.split('.')[:-1]))
        # it if fails, try to load from up tree directory (I am prediction/core/preprocessing/preprocess.py)
        except ImportError:
            try:
                py_module = importlib.import_module("....preprocessing" + preproc_name, '.'.join(__name__.split('.')[:-1]))
            # it if fails, try to laod from module_path.core
            except ImportError:
                py_module = importlib.import_module(module_path + ".core.preprocessing" + preproc_name,
                                                         '.'.join(__name__.split('.')[:-1]))

        preproc_module = getattr(py_module, preproc_name)(**preproc_kwargs)

    except Exception as e:
        raise Exception("problem with module: %s, kwargs: %s, exception %s" % (preproc_name, preproc_kwargs, e)) from e

    return preproc_module
