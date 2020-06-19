import importlib

def get_attack_class(method_name, module_path=""):
    try:
        # # first try to load from here
        # try:
        py_module = importlib.import_module("." + method_name, '.'.join(__name__.split('.')[:-1]))

        # # it if fails, try to load from up tree directory
        # except ImportError:
        #     try:
        #         py_module = importlib.import_module("....transform" + transform_name, '.'.join(__name__.split('.')[:-1]))
        #     # it if fails, try to laod from module_path.core
        #     except ImportError:
        #         py_module = importlib.import_module(module_path + ".core.transform" + transform_name,
        #                                                  '.'.join(__name__.split('.')[:-1]))

        attack_class = getattr(py_module, method_name)

    except Exception as e:
        raise Exception("problem with module: %s, exception %s" % (method_name, e)) from e

    return attack_class


name_short = {
    'CarliniWagner': 'CW',
    'EOT': 'EOT',
    'EOT_CarliniWagner': 'EOT_CW',
    'BasicIterativeMethod': 'BIM'
}


def get_attack_id(method_name, method_kwargs):
    """Creates the id for an attack.

    Args:
        method_tuple (tuple): A tuple composed of : (name of the builder function, kwargs to pass to the function).

    Returns:
        string: the idname of the function that we want to concatenate in the output filenames.

    """

    # listWithPoints = lambda x: ".".join(re.sub('[( )\[\]]', '', str(x)).replace(' ', '').split(","))

    methodid = name_short[method_name]

    if method_name == 'CarliniWagner':
        methodid += "_n" + "{:d}".format(method_kwargs['num_steps'])
        methodid += "_po" + "{:}".format(maybe_capitalize(method_kwargs['proj_ord']))
        methodid += "_do" + "{:}".format(maybe_capitalize(method_kwargs['ldist_ord']))
        # methodid += "_c" + str(int(method_kwargs['const']))
        methodid += "_lr" + "{:.2e}".format(method_kwargs['learning_rate'])
    elif method_name == 'EOT':
        methodid += "_n" + "{:d}".format(method_kwargs['num_steps'])
        methodid += "_ss" + "{:d}".format(method_kwargs['sample_size'])
        methodid += "_lr" + "{:.2e}".format(method_kwargs['learning_rate'])
        methodid += "_po" + "{:}".format(maybe_capitalize(method_kwargs['proj_ord']))
    elif method_name == 'EOT_CarliniWagner':
        methodid += "_n" + "{:d}".format(method_kwargs['num_steps'])
        methodid += "_po" + "{:}".format(maybe_capitalize(method_kwargs['proj_ord']))
        methodid += "_do" + "{:}".format(maybe_capitalize(method_kwargs['ldist_ord']))
        methodid += "_lr" + "{:.2e}".format(method_kwargs['learning_rate'])
        methodid += "_ss" + "{:d}".format(method_kwargs['sample_size'])
    elif method_name == 'BasicIterativeMethod':
        methodid += "_n" + "{:d}".format(method_kwargs['num_steps'])
        methodid += "_po" + "{:}".format(maybe_capitalize(method_kwargs['proj_ord']))
        methodid += "_lr" + "{:.2e}".format(method_kwargs['learning_rate'])
    else:
        print('----------------------')
        print('ERROR ', method_name)
        raise ValueError("id rule for `%s` has to be implemented." % method_name)

    return methodid

def maybe_capitalize(ord_string):
    if ord_string == "inf":
        ord_string = "Inf"

    return ord_string
