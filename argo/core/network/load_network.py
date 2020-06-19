import importlib

from argo.core.utils.argo_utils import eval_method_from_tuple


def instantiate_network(opts, name, module_path=""):
    network_name = opts.get("network", "FFNetwork")

    try:
        # first try to load from core
        try:
            network_module = importlib.import_module("core." + network_name, '.'.join(__name__.split('.')[:-1]))
        # it if fails, try to load from up tree directory (I am core/argo/core/network/Network.py)
        except ImportError:
            try:
                network_module = importlib.import_module("...." + network_name, '.'.join(__name__.split('.')[:-1]))
            # it if fails, try to laod from module_path.core
            except ImportError:
                network_module = importlib.import_module(module_path + ".core." + network_name,
                                                         '.'.join(__name__.split('.')[:-1]))

        network_tuple = (network_name, {
            "opts": opts,
            "name": name})
        network = eval_method_from_tuple(network_module, network_tuple)

    except Exception as e:
        raise Exception("problem in 'instantiate_network' with module: %s, kwargs: %s, exception %s" % (network_name, opts, e)) from e

    return network
