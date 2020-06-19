import os
import importlib
from core.argo.core.ArgoLauncher import ArgoLauncher
from core.argo.core.TFDeepLearningModel import load_network
from core.argo.core.utils.argo_utils import load_class
from datasets.Dataset import Dataset
import numpy as np
from abc import abstractmethod


def get_transform_module(transform_name, transform_kwargs, module_path=""):

    try:
        # # first try to load from here
        # try:
        py_module = importlib.import_module("." + transform_name, '.'.join(__name__.split('.')[:-1]))

        # # it if fails, try to load from up tree directory
        # except ImportError:
        #     try:
        #         py_module = importlib.import_module("....transform" + transform_name, '.'.join(__name__.split('.')[:-1]))
        #     # it if fails, try to laod from module_path.core
        #     except ImportError:
        #         py_module = importlib.import_module(module_path + ".core.transform" + transform_name,
        #                                                  '.'.join(__name__.split('.')[:-1]))

        #import pdb;pdb.set_trace()

        transform_module = getattr(py_module, transform_name)(**transform_kwargs)

    except Exception as e:
        raise Exception("problem with module: %s, kwargs: %s, exception %s" % (transform_name, transform_kwargs, e)) from e

    return transform_module


def load_network_with_dummy_x(dummy_x, sess, is_training, conf_file, global_step, base_path):
    dataset_conf, model_parameters, config = ArgoLauncher.process_conf_file(conf_file)
    dataset = Dataset.load_dataset(dataset_conf)
    check_dataset_shapes(dataset.x_shape_eval, dummy_x.shape[1:])

    full_class_path = base_path + "." + model_parameters["model"]

    model_class = load_class(full_class_path)
    network, checkpoint_name = load_network(model_class,
                                          conf_file=conf_file,
                                          dataset=dataset,
                                          global_step=global_step)
    # LOAD NETWORK

    stuff = network(dummy_x, is_training = is_training)
    network.restore(sess, checkpoint_name)
    return network


def check_dataset_shapes(shape1, shape2):
    # if datasets x_shape are different raise Exception! what is the meaning of a comparison otherwise?
    np.testing.assert_array_equal(shape1, shape2,
                "the datasets that you are trying to load have \
                different x_shape : `%s` and `%s`" % (
                str(shape1), str(shape2))
                )



def get_transform_id(method_name, method_kwargs):
    """Creates the id for a transformation.

    Args:
        method_tuple (tuple): A tuple composed of : (name of the builder function, kwargs to pass to the function).

    Returns:
        string: the idname of the function that we want to concatenate in the output filenames.

    """

    # listWithPoints = lambda x: ".".join(re.sub('[( )\[\]]', '', str(x)).replace(' ', '').split(","))

    if method_name == 'identity':
        methodid = "clean"

    # elif method_name == 'resize':
    #     methodid = 'resize' + method_kwargs['intermediate_size']

    elif method_name == 'ae':
        methodid = extract_last_dir(method_kwargs['conf_file'])

        if method_kwargs['global_step'] is not None:
            methodid += "-gs" + method_kwargs['global_step']
        methodid += "-sh" #+ "{:d}".format(method_kwargs["sample_hid"])

    elif method_name == 'vae':
        methodid = extract_last_dir(method_kwargs['conf_file'])

        if method_kwargs['global_step'] is not None:
            methodid += "-gs" + method_kwargs['global_step']

        methodid += "-sh" + "{:d}".format(method_kwargs["sample_hid"])
        methodid += "-sx" + "{:d}".format(method_kwargs["sample_vis"])
        methodid += "-zsc" + str(method_kwargs["z_std_scale"])


    else:
        print('----------------------')
        print('ERROR ', method_name)
        raise ValueError("id rule for `%s` has to be implemented." % method_name)

    return methodid

def extract_last_dir(path):
    return os.path.basename(os.path.dirname(path))


# maybe needed if transformations gets more complicated
class Transformation:
    default_params_for_attack = {}

    @abstractmethod
    def build(self, **kwargs):
        """
        build the transform method

        Args:
            **kwargs: arguments for the transform builder

        Returns:
            The transform method

        """
        pass
