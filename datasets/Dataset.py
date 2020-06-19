""" Dataset Module

Dataset module introduces an abstract class that is the skeleton of all the
datasets classes. It provides the methods that Argo use to feed our ML
algorithms.

"""

from .PerturbedDecorator import PerturbedDecorator

from abc import ABC, abstractmethod
import importlib
from functools import partial
import tensorflow as tf

import numpy as np

import pdb

from . import Augmentations
from . import Perturbations

TRAIN_LOOP = "train_loop"
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
TRAIN_SHUFFLED = "train_shuffled"
VALIDATION_SHUFFLED = "validation_shuffled"
TEST_SHUFFLED = "test_shuffled"

NPROCS = 20

short_name_dataset = {
    TRAIN_LOOP : "tr_loop",
    TRAIN : "train",
    VALIDATION : "val",
    TEST : "test",
    TRAIN_SHUFFLED : "train_sh",
    VALIDATION_SHUFFLED : "val_sh",
    TEST_SHUFFLED : "test_sh"
}

linestyle_dataset = {
    TRAIN_LOOP : "-",
    TRAIN : ":",
    VALIDATION : "--",
    TEST : "-.",
    TRAIN_SHUFFLED : ":",
    VALIDATION_SHUFFLED : "--",
    TEST_SHUFFLED : "-."
}


def check_dataset_keys_not_loop(datasets_keys):
    allowed_keys_list = [TRAIN, VALIDATION, TEST, TRAIN_SHUFFLED, VALIDATION_SHUFFLED, TEST_SHUFFLED]
    if not isinstance(datasets_keys, list):
        datasets_keys = [datasets_keys]

    for ds_key in datasets_keys:
        if ds_key not in allowed_keys_list:
            raise ValueError("specified string '%s' is not allowed, provide one of: %s " % (str(ds_key) , str(allowed_keys_list)))


def load_class(module_plus_class):
    """
    from a module  import and return a class

    Parameters
    ----------
    module_plus_class : str
        in the form "Module.Class"

    Returns
    -------
    class
        Returns the class

    """

    my_module = importlib.import_module(".".join(module_plus_class.split(".")[:-1]))
    my_class = getattr(my_module, module_plus_class.split(".")[-1])
    return my_class


class Dataset(ABC):
    """ Abstract class for the dataset objects

        the static method load_dataset is a factory method that
        chose the correct child. Child inherit

    """

    implemented_params_keys = ['dataName']  #: all the keys properties available in the dataset

    default_params = {}

    classification = None   # bool, true if the object y is for classification
                            # it abilitates classes.

    @staticmethod
    def load_dataset(params, return_params_with_default_values=False, base_path=''):
        """
            return the dataset object

            Parameters
            ----------
            params : dict
                the config dictionary that must contain the key "dataName"
                which selects the dataset and the dataset object, e.g.:
                 - 'MNIST' - is the standard DL dataset
                 - 'cifar10' -
                 - 'boston'  - Boston house dataset, for regression tasks

            return_params_with_default_values : boole
                if true returns also the updated params with the default values

            Returns
            -------
            dataset
                the object

            params_with_default_values (optional)
                updated params with default values

        """

        if not base_path=='':
            base_path+="."

        try:
            # load the dataset class that should be in the module dataName.py
            dataset_class = load_class(base_path+"datasets."+params["dataName"]+"."+params["dataName"])
        except:
            raise Exception('Dataset "{}" not recognized or bugged'.format(params["dataName"]))

        # add default values to params
        params_with_default_values = dataset_class.add_default_parameters(params)
        params = params_with_default_values


        # TODO: COMMENT IT AND ADD IT TO THE DOCSTRING
        if "perturbed" in params and params["perturbed"] == 1:

            if "perturbed_id" not in params:
                raise Exception("You are supposed to specify a new perturbed_id, since this is supposed to be an perturbed dataset")

            decorator = PerturbedDecorator(params["perturbed_id"])
            if "perturbed_train_set_x" in params.keys():
                decorator.train_set_x_fileName = params["perturbed_train_set_x"]
            if "perturbed_test_set_x" in params.keys():
                decorator.test_set_x_fileName = params["perturbed_test_set_x"]
            # decore the class
            dataset_class = decorator(dataset_class)

        #import pdb;pdb.set_trace()
        dataset_object = dataset_class(params)

        if return_params_with_default_values:
            return params_with_default_values, dataset_object
        else:
            return dataset_object

    def __init__(self, params):
        # provides default options if not already specified
        self._params_original = params.copy()
        params.update(self.default_params)
        params.update(self._params_original)

        self._id = self.dataset_id(params)
        self._params = params

        self._labels = 0
        self._n_labels = 0

        self._data_augmentation = params.get("dataAugmentation", [])
        self._data_perturbation = params.get("dataPerturbation", [])
        
        self._binary_input = False
        self._train_set_x = None
        self._train_set_y = None
        self._validation_set_x = None
        self._validation_set_y = None
        self._test_set_x = None
        self._test_set_y = None

        self._caching_bool = True
        self._shuffling_cache = None

    @staticmethod
    def get_default_params():
        """ returns the dictionary with default values"""
        return Dataset.default_params

#TODO why add_default_parameters and process_params seems to do the same thing?
    @classmethod
    def add_default_parameters(cls, params):
        """ append to the dictionary params the default values if not present """
        def_values = cls.default_params.copy()
        def_values.update(params)

        return def_values

    @classmethod
    def process_params(cls, params):
        """
        Preprocessing of the parameters from config and argo.

        In practice it updates the default values with the ones produced by
        argo from the config file.

        Parameters
        ----------
        params : dict
            parameters extreacted from the config file

        Returns
        -------
        full_params : dict
            the params dictionary where missing values are replace by default
            ones.

        """

        full_params = cls.default_params.copy()

        # check the dimension
        # if len(full_params) + 2 < len(params): # run are introduced dynamically
        #    print("Warning: argo passed more options then those in default")

        # update the params dictionary
        full_params.update(params)

        # # unpack full_params
        # for k in full_params.keys():
        #     if type(full_params[k]) == list:
        #         if len(full_params[k]) != 1:
        #             raise Exception(
        #                 'The option {} is not a single valued list'.format(k))
        #         else:
        #             full_params[k] = full_params[k][0]

        return full_params

#TODO id should not be a static method to be able to have multilevel handling of the parameters,
#TODO see how this is done in Models and Networks
    @staticmethod
    @abstractmethod
    def dataset_id(self, params):
        """ return the id once processed the parameters in params """
        #ids, parameters = Dataset.process_params(params)
        #return ids
        pass

    @classmethod
    def check_params_impl(cls, params):
        """check if there are parameters that have not been implemented yet"""
        # for prop in params.keys():
        #     if prop not in cls.implemented_params_keys:
        #         print('[Warning] - the param {} - is not yet implemented'\
        #               ' in the dataset {}'.format(prop, str(cls)))
        pass

    @staticmethod
    def class_filter(dataset_x, dataset_y, classes, position_label):
        """
        return the dataset with labels in the list classes

        :param dataset_x: data
        :param dataset_y: labels
        :param classes:    list of classes
        :param position_label:  list of classes
        :return: (dataset_x, dataset_y) with filtered elemnts not in classes
        """

        ix_match_class_dataset = np.in1d(dataset_y, classes)
        dataset_x = dataset_x[ix_match_class_dataset]
        dataset_y = dataset_y[ix_match_class_dataset]

        if position_label:

            def replace_with_position(label_set, classes):
                label_set_new = np.copy(label_set)
                for ix, class_ in enumerate(classes): label_set_new[label_set == class_] = ix
                return label_set_new

            dataset_y = replace_with_position(dataset_y, classes)

        return dataset_x, dataset_y

    # this function can be used in case I want to lazy load dinamically the files (see CMB/BRATS)
    # or certain type of data augmentation (see MNIST)
    def dataset_map(self, dataset, datasets_tuple):
        output_types = self.get_output_types(datasets_tuple)
       
        def get_function(n):
            return full_data[n]

        def apply_py_function(n):
            #if isinstance(output_types[0], list):
            #    #pdb.set_trace()
            #    #return (tf.py_func(get_function, [n], output_types[0]),) + tuple(tf.py_func(get_function, [n], output_types[1:]))
            #    return tuple(tf.py_func(get_function, [n], output_types))
            #else:
            #    pdb.set_trace()
            return tuple(tf.py_func(get_function, [n], output_types))

        #if isinstance(output_types[0], list):
        #    full_data = list(zip(tuple([a, b] for a,b in zip(*datasets_tuple[0])), *datasets_tuple[1:]))
        #else:
        #    # I need a list of iterators
        full_data = list(zip(*datasets_tuple))

        #dataset = dataset.map(
        #    lambda n: tuple(tf.py_func(get_function, [n], output_types)),
        #    num_parallel_calls=NPROCS)


        dataset = dataset.map(apply_py_function, num_parallel_calls=NPROCS)
                        
        return dataset

    def get_output_shapes(self, datasets_tuple):
        #if not isinstance(datasets_tuple[0], list):
        output_shapes = tuple(ds[0].shape for ds in datasets_tuple)
        #else:
        #    output_shapes = ([ds[0].shape for ds in datasets_tuple[0]], ) + tuple(ds[0].shape for ds in datasets_tuple[1:])

        return output_shapes

    def get_output_types(self, datasets_tuple):    
        output_types = tuple([tf.as_dtype(ds[0].dtype) for ds in datasets_tuple])

        #if not isinstance(datasets_tuple[0], list):
        #    output_types = tuple(tf.as_dtype(ds[0].dtype) for ds in datasets_tuple)
        #else:
        #    output_types = ([tf.as_dtype(ds[0].dtype) for ds in datasets_tuple[0]], ) + tuple(tf.as_dtype(ds[0].dtype) for ds in datasets_tuple[1:])
        #    #output_types = (tf.as_dtype(datasets_tuple[0][0].dtype), ) + tuple(tf.as_dtype(ds[0].dtype) for ds in datasets_tuple[1:])

        return output_types

    def get_dataset_iterator(self, batch_size, dataset_str, shuffle, repeat, augment, perturb):
        #is_perturbed = False
        datasets_tuple = None

        # create datasets_tuple objects
        # if is_perturbed, then I return ((x, x_perturbed), y)
        if dataset_str == TRAIN:
            datasets_tuple = (self.train_set_x, )
            #if len(self._data_perturbation)>0:
            #    datasets_tuple = ([datasets_tuple[0], self.train_set_x] ,)                
            '''
            if hasattr(self,"perturbed_train_set_x") and self.perturbed_train_set_x is not None:
                datasets_tuple = (datasets_tuple + (self.perturbed_train_set_x, ) ,)
                is_perturbed = True
                raise Exception("'perturbed_train_set_x' is now obsolte, talk to Luigi")
            '''
            if self._train_set_y is not None:
                datasets_tuple = datasets_tuple + (self.train_set_y, )

        elif dataset_str == VALIDATION:
            datasets_tuple = (self.validation_set_x, )
            #if len(self._data_perturbation)>0:
            #    datasets_tuple = (datasets_tuple + (self.validation_set_x, ) ,)

            '''
            if hasattr(self,"perturbed_validation_set_x") and self.perturbed_validation_set_x is not None:
                datasets_tuple = (datasets_tuple + (self.perturbed_validation_set_x, ) , )
                is_perturbed = True
                raise Exception("'perturbed_validation_set_x' is now obsolte, talk to Luigi")
            '''
            if self._validation_set_y is not None:
                datasets_tuple = datasets_tuple + (self.validation_set_y, )

        elif dataset_str == TEST:
            datasets_tuple = (self.test_set_x, )
            #if len(self._data_perturbation)>0:
            #    datasets_tuple = (datasets_tuple + (self.test_set_x, ) ,)

            '''
            if hasattr(self,"perturbed_test_set_x") and self.perturbed_test_set_x is not None:
                datasets_tuple = (datasets_tuple + (self.perturbed_test_set_x, ), )
                is_perturbed = True
                raise Exception("'perturbed_test_set_x' is now obsolte, talk to Luigi")
            '''
            if self._test_set_y is not None:
                datasets_tuple = datasets_tuple + (self.test_set_y, )

        else:
            raise Exception("dataset not recognized (accepted values are: train, validation and test)")

        if isinstance(datasets_tuple[0], list):
            raise Exception("not supported")
            n_samples = datasets_tuple[0][0].shape[0]
        else:
            n_samples = datasets_tuple[0].shape[0]
        # creates a dataset of step separated range of values.
        dataset = tf.data.Dataset.range(n_samples)

        #print(n_samples)
        
        output_shapes = self.get_output_shapes(datasets_tuple)
        
        #print(output_shapes)
        
        dataset = self.dataset_map(dataset, datasets_tuple)
        
        # do not add "_" or this will raise issues when loading the model
        def set_shapes(*nodes):
            for node, outshape in zip(nodes, output_shapes):
                if isinstance(node, tuple):
                    for n, os in zip(node, output_shape):
                        n.set_shape(os)
                else:
                    node.set_shape(outshape)
            return nodes

        dataset = dataset.map(set_shapes, num_parallel_calls=NPROCS)

        # caching before shuffling and batching for super cow speed
        # Luigi: super cow speed? what do you mean?
        if self._caching_bool:
            dataset = dataset.cache()
            
        # data augmentation
        if augment:
            # the line below is more elegant, however not compatible with loading PBs (Luigi)
            #dataset = dataset.map(partial(self.augment_element, is_perturbed), num_parallel_calls=NPROCS)
            dataset = dataset.map(self.augment_element, num_parallel_calls=NPROCS)
        else:
            dataset = dataset.map(self.duplicate_x_element_if_needed, num_parallel_calls=NPROCS)
        
        # make sure I have some perturbation, otherwise I don't duplicate the tensor
        if perturb and len(self._data_perturbation)>0:
            #if perturb:
            dataset = dataset.map(self.perturb_element, num_parallel_calls=NPROCS)
        else:
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
            #print(dataset)
            dataset = dataset.map(self.duplicate_x_element_if_needed, num_parallel_calls=NPROCS)
            #print(dataset)
            #pdb.set_trace()
        
        
        # we shuffle the data and sample repeatedly batches for training
        if shuffle:
            if self._shuffling_cache is None:
                shuffling_cache = n_samples + 1
            else:
                shuffling_cache = self._shuffling_cache

            dataset = dataset.shuffle(shuffling_cache)

        if repeat:
            dataset = dataset.repeat()
            batched_dataset = dataset.batch(batch_size)
            # create iterator to retrieve batches
            iterator = batched_dataset.make_one_shot_iterator()
            # initializer = None
        else:
            batched_dataset = dataset.batch(batch_size)
            iterator = batched_dataset.make_initializable_iterator()
            # initializer = iterator.initializer

        # batched_dataset = batched_dataset.prefetch(500)

        # get a training batch of images and labels
        return iterator #, is_perturbed

    def perturb_element(self, *observation):

        nargs = len(observation)

        raw_x = observation[0][0]
        aug_x = observation[0][1]
        perturb_x = aug_x
 
        #if isinstance(observation[0], tuple):
        #    x = observation[0][0]
        #    x_perturbed = observation[0][1]
        #else:
        #    x = observation[0]
        #    x_perturbed = x
            
        if nargs == 2:
            y = observation[1]
        elif nargs>2:
            raise Exception("observation of the dataset is a tuple with more than 2 elements.")

        def apply_perturbation(function, kwargs_tuple, x):
            return function(x, **kwargs_tuple)

        for pert in self._data_perturbation:
        
            method, kwargs_tuple, frequency_perturbation = pert
            function = getattr(Perturbations, method.split(".")[-1])

            # determining if I will apply the perturbation
            uniform_sample = tf.random_uniform([], 0., 1., dtype=tf.float32) #shape, min, max

            if frequency_perturbation > 0.0:
                cond = tf.less(uniform_sample, frequency_perturbation)
                
                perturb_x = tf.cond(cond,
                                      lambda: apply_perturbation(function,
                                                                 kwargs_tuple,
                                                                 perturb_x),
                                      lambda: perturb_x)
                
        perturbed_observation = ((raw_x, aug_x, perturb_x), )
        
        if y is not None:
            perturbed_observation += (y,)

        return perturbed_observation

    # I expect ((raw_x, aug_x)) or ((raw_x, aug_x), y)
    # I return ((raw_x, aug_x, perturb_x)) or ((raw_x, aug_x, perturbed_x), y)
    # with perturb_x = aug_x
    def duplicate_x_element_if_needed(self, *observation):

        nargs = len(observation)
        y = None
        
        current_x = observation[0]
        if isinstance(current_x, tuple):
            duplicated_x = current_x[-1]
        else:
            duplicated_x = current_x
            current_x = (current_x,)
            
        #if isinstance(observation[0], tuple):
        #    x = observation[0][0]
        #    x_perturbed = observation[0][1]
        #else:
        #    x = observation[0]
        #    x_perturbed = x
            
        if nargs == 2:
            y = observation[1]
        elif nargs>2:
            raise Exception("observation of the dataset is a tuple with more than 2 elements.")

        perturbed_observation = (current_x + (duplicated_x, ), )
        
        if y is not None:
            perturbed_observation += (y,)

        # print(perturbed_observation)
        #if isinstance(observation[0], tuple):
        #    pdb.set_trace()
        return perturbed_observation

    # The function expects x or (x,y) and (x_raw, x_aug)) or ((x_raw, x_aug), y)
    def augment_element(self, *observation):
        nargs = len(observation)
        y = None
        x_perturbed = None

        assert nargs<=2, "The function expects x or (x,y) in input"
        
        raw_x = observation[0]
        aug_x = raw_x
        if nargs == 2:
            y = observation[1]

        def apply_augmentation(function, kwargs_tuple, x):
           # if x_perturbed is None:
           x = function(x, **kwargs_tuple)
           return x
           # else:
           #     xs = function([x, x_perturbed], **kwargs_tuple)
           #     return xs

        for aug in self._data_augmentation:
        
            method, kwargs_tuple, frequency_augmentation = aug
            function = getattr(Augmentations, method.split(".")[-1])

            # determining if I will apply the data augmentation
            uniform_sample = tf.random_uniform([], 0., 1., dtype=tf.float32) #shape, min, max

            if frequency_augmentation > 0.0:
                cond = tf.less(uniform_sample, frequency_augmentation)
                #if x_perturbed is None:
                aug_x = tf.cond(cond,
                                lambda: apply_augmentation(function,
                                                           kwargs_tuple,
                                                           aug_x),
                                lambda: aug_x)
                # this has not been tested
                #else:
                #    (x, x_perturbed) = tf.cond(cond,
                #                               lambda: apply_augmentation(function,
                #                                                          kwargs_tuple,
                #                                                          x,
                #                                                          x_perturbed),
                #                               lambda: (x, x_perturbed))

        #if x_perturbed is None:
        #    augmented_observation = (x,)
        #else:
        #    augmented_observation = ((x, x_perturbed), )

        augmented_observation = ((raw_x, aug_x), )

        if y is not None:
            augmented_observation += (y,)

        #print(augmented_observation)

        return augmented_observation
    
    '''
    # NOT USED ANYMORE SINCE NOT COMPATIBLE WITH PB FILES (Luigi)
    # do not add "_" or this will raise issues when loading the model
    def augment_element(self, is_perturbed, *observation):
        #perform preprocessing to the observation of an element of the dataset,
        #optionally augment the observation
        #(no random noise, this part will be handled at the model level for the moment)
        #
        #Args:
        #    is_perturbed (bool): if it is True, we expect observation = (x, x_perturbed, [y]). Where y is optional
        #                        if it is False, we expect observation = (x, [y]). Where y is optional
        #    *observation (list of tf.Tensor): a list of tensors.
        #
        #Returns:
        #    type: Description of returned object.
        #

        pdb.set_trace()

        nargs = len(observation)
        y = None
        x_perturbed = None

        if is_perturbed:
            x = observation[0]
            x_perturbed = observation[1]
            if nargs == 3:
                y = observation[2]
            elif nargs>3:
                raise Exception("observation of the dataset is a tuple with more than 3 elements.\
                                But it is perturbed and it should be of length either 2 or 3")
        else:
            x = observation[0]
            if nargs == 2:
                y = observation[1]
            elif nargs>2:
                raise Exception("observation of the dataset is a tuple with more than 2 elements.\
                                But it is not perturbed and it should be of length either 1 or 2")

        def apply_augmentation(function, kwargs_tuple, x, x_perturbed = None):
            if x_perturbed is None:
                x = function(x, **kwargs_tuple)
                return x
            else:
                xs = function([x, x_perturbed], **kwargs_tuple)
                return xs

        for aug in self._data_augmentation:
        
            method, kwargs_tuple, frequency_augmentation = aug
            function = getattr(Augmentations, method.split(".")[-1])

            # determining if I will apply the data augmentation
            uniform_sample = tf.random_uniform([], 0., 1., dtype=tf.float32) #shape, min, max

            if frequency_augmentation > 0.0:
                cond = tf.less(uniform_sample, frequency_augmentation)
                if x_perturbed is None:
                    x = tf.cond(cond,
                                lambda: apply_augmentation(function,
                                                           kwargs_tuple,
                                                           x),
                                lambda: x)
                # this has not been tested
                else:
                    (x, x_perturbed) = tf.cond(cond,
                                               lambda: apply_augmentation(function,
                                                                          kwargs_tuple,
                                                                          x,
                                                                          x_perturbed),
                                               lambda: (x, x_perturbed))

        augmented_observation = (x,)
        if x_perturbed is not None:
            augmented_observation += (x_perturbed,)

        if y is not None:
            augmented_observation += (y,)

        return augmented_observation
    '''

    def get_dataset_with_handle(self, batch_size_train, batch_size_eval):

        datasets = []
        #perturbed_bools = []

        # create iterators for all datasets
        train_loop_iterator = self.get_dataset_iterator(batch_size_train, "train",
                                                           shuffle = 1,
                                                           repeat = 1,
                                                           augment = 1,
                                                           perturb = 1)
        datasets.append(train_loop_iterator)
        # no need to add this
        #perturbed_bools.append(train_loop_is_perturbed)

        train_iterator = self.get_dataset_iterator(batch_size_eval, "train",
                                                   shuffle = 0,
                                                   repeat = 0,
                                                   augment = 1,
                                                   perturb = 1)
        datasets.append(train_iterator)
        #perturbed_bools.append(train_is_perturbed)

        validation_iterator = self.get_dataset_iterator(batch_size_eval, "validation",
                                                        shuffle = 0,
                                                        repeat = 0,
                                                        augment = 1,
                                                        perturb = 1)
        datasets.append(validation_iterator)
        #perturbed_bools.append(validation_is_perturbed)

        test_iterator = self.get_dataset_iterator(batch_size_eval, "test",
                                                  shuffle = 0,
                                                  repeat = 0,
                                                  augment = 0,
                                                  perturb = 0)
        datasets.append(test_iterator)
        #perturbed_bools.append(test_is_perturbed)

        # create sheffled iterators
        train_shuffled_iterator = self.get_dataset_iterator(batch_size_eval, "train",
                                                            shuffle = 1,
                                                            repeat = 0,
                                                            augment = 1,
                                                            perturb = 1)
        datasets.append(train_shuffled_iterator)
        #perturbed_bools.append(train_shuffled_is_perturbed)
        validation_shuffled_iterator = self.get_dataset_iterator(batch_size_eval, "validation",
                                                                 shuffle = 1,
                                                                 repeat = 0,
                                                                 augment = 1,
                                                                 perturb = 1)
        datasets.append(validation_shuffled_iterator)
        #perturbed_bools.append(validation_shuffled_is_perturbed)

        test_shuffled_iterator  = self.get_dataset_iterator(batch_size_eval, "test",
                                                            shuffle = 1,
                                                            repeat = 0,
                                                            augment = 0,
                                                            perturb = 0)
        datasets.append(test_shuffled_iterator)
        #perturbed_bools.append(test_shuffled_is_perturbed)

        '''
        if all(perturbed_bools):
            is_perturbed = True
        elif not any(perturbed_bools):
            is_perturbed = False
        else:
            raise Exception("datasets must be all perturbed or none perturbed. Found %s"%str(perturbed_bools))
        '''
        
        handle = tf.placeholder(tf.string, shape=[])

        #TODO check that output_types match
        agreed_types = train_iterator.output_types
        # I will need to check that all the datasets have the same output_shapes,
        # otherwise if they do not agree on a dimension it will be put at None
        # (we want to allow for variable input shapes if the networks permit this, e.g. a fully convolutional) (Riccardo)

        agreed_shapes = []

        shapes_ref_list = datasets[0].output_shapes
        shapes_equals_list = []
                
        for d in datasets:
            shapes_list = d.output_shapes
            shapes_equals = []
            for shape, shape_ref in zip(shapes_list, shapes_ref_list):
                # element-wise equality
                shapes_equals.append(np.equal(shape, shape_ref))
                #if isinstance(shape, tuple):
                #    pdb.set_trace()
                #    shapes_equals.append(np.equal([s.as_list() for s in shape], [s.as_list() for s in shape_ref]))
                #else:
                #    shapes_equals.append(np.equal(shape.as_list(), shape_ref.as_list()))
            shapes_equals_list.append(shapes_equals)

        for i_s, bools in enumerate(zip(*shapes_equals_list)):
            shape_ref = shapes_ref_list[i_s]
            # element-dim_boolswise
            #pdb.set_trace()
            dim_bools = np.all(bools, axis=0)

            if isinstance(shape_ref, tuple):
                new_shape = []
                for sf, db in zip(shape_ref, dim_bools):
                    ns = []
                    for dim, bool in zip(sf, db):
                        #pdb.set_trace()
                        if bool:
                            ns.append(dim)
                        else:
                            ns.append(None)

                    new_shape.append(ns)
                
                agreed_shapes.append(tuple(new_shape))
            else:
                new_shape = []
                for dim, bool in zip(shape_ref, dim_bools):
                    #pdb.set_trace()
                    if bool:
                        new_shape.append(dim)
                    else:
                        new_shape.append(None)

                agreed_shapes.append(new_shape)

        agreed_shapes = tuple(agreed_shapes)
        
        # create general iterator from handle
        iterator = tf.data.Iterator.from_string_handle(handle, agreed_types, agreed_shapes)

        batch_x = iterator.get_next()

        ds_iterators = { TRAIN_LOOP : train_loop_iterator,
                         TRAIN : train_iterator,
                         VALIDATION : validation_iterator,
                         TEST : test_iterator,
                         TRAIN_SHUFFLED : train_shuffled_iterator,
                         VALIDATION_SHUFFLED : validation_shuffled_iterator,
                         TEST_SHUFFLED : test_shuffled_iterator}

        ds_initializers = { key : iterator.initializer for (key, iterator) in ds_iterators.items() if key != TRAIN_LOOP}
        ds_handles = { key : iterator.string_handle() for (key, iterator) in ds_iterators.items()}

        return batch_x, handle, ds_initializers, ds_handles #, is_perturbed


    # agree to deprecate it but please let's leave it like this just to quickly play around for simple datasets (?)
    # deprecated, to be removed
    def get_raw_elements(self, dataset_str, index_list=None):
        check_dataset_keys_not_loop(dataset_str)
        attribute_name = dataset_str + "_set_x"
        if index_list is not None:
            x_data = getattr(self, attribute_name)[index_list]
        else:
            x_data = getattr(self, attribute_name)
        return x_data

    # agree to deprecate it but please let's leave it like this just to quickly play around for simple datasets (?)
    # deprecated, to be removed
    def get_raw_labels(self, dataset_str, index_list=None):
        check_dataset_keys_not_loop(dataset_str)
        attribute_name = dataset_str+"_set_y"
        if index_list is not None:
            y_data = getattr(self, attribute_name)[index_list]
        else:
            y_data = getattr(self, attribute_name)

        return y_data

    # new function
    def get_elements(self, node, ds_handle_node, ds_handle_value, ds_initializer, session, index_list=None):
        session.run(ds_initializer)

        return_list = []
        while index_list==None or len(return_list)<=np.max(index_list):
            try:
                r = session.run([node], feed_dict = {ds_handle_node : ds_handle_value})

                if len(return_list)==0:
                    return_list = np.array(r)[0]
                else:
                    return_list = np.concatenate((return_list, np.array(r)[0]), axis=0)
            except tf.errors.OutOfRangeError:
                break

        if index_list is None:
            return return_list
        else:
            return return_list[index_list]

    @property
    def id(self):
        """
        Returns th id of the Dataset

        :return: the id
        """
        return self._id

    @property
    def binary_input(self):
        return self._binary_input

    def n_samples(self, dataset_str):
        check_dataset_keys_not_loop(dataset_str)
        attribute_name = "n_samples_" + dataset_str
        n = getattr(self, attribute_name)
        return n

    @property
    def n_samples_train(self):
        return len(self._train_set_x)

    @property
    def n_samples_validation(self):
        if self._validation_set_x is None:
            return 0
        else:
            return len(self._validation_set_x)

    @property
    def n_samples_test(self):
        if self._test_set_x is None:
            return 0
        else:
            return len(self._test_set_x)

    @property
    def train_set_x(self):
        return self._train_set_x

    @property
    def train_set_y(self):
        if self._train_set_y is None:
            raise ValueError("Labels not available")
        else:
            return self._train_set_y

    @property
    def validation_set_x(self):
        if self._validation_set_x is None:
            raise Exception("Validation set not available")
        else:
            return self._validation_set_x

    @property
    def validation_set_y(self):
        if self._validation_set_x is None:
            raise Exception("Validation set not available")
        else:
            if self._validation_set_y is None:
                raise Exception("Labels not available")
            else:
                return self._validation_set_y

    @property
    def test_set_x(self):
        if self._test_set_x is None:
            raise Exception("Test set not available")
        else:
            return self._test_set_x

    @property
    def test_set_y(self):
        if self._test_set_x is None:
            raise Exception("Test set not available")
        else:
            if self._test_set_y is None:
                raise ValueError("Labels not available")
            else:
                return self._test_set_y

    @property
    def x_shape(self):
        """return the shape of an input sample for the train loop"""
        return self.x_shape_train

    @property
    def x_shape_train(self):
        """return the shape of an input sample for the train loop"""
        return self.train_set_x[0].shape


    @property
    def x_shape_eval(self):
        """return the shape of an input sample for evaluation"""
        return self.train_set_x[0].shape

    @property
    def y_shape(self):
        """return the shape of an output sample"""
        train_set_y = getattr(self, 'train_set_y', None)
        if train_set_y is None:
            raise ValueError("this dataset does not have y set")

        return train_set_y[0].shape

    def set_labels_attr(self):
        """
        compute and set the labels attributes

        in particular  self.labels, self.n_labels
         """
        self._labels = np.unique(self.train_set_y)
        self._n_labels = len(self._labels)

    @property
    def n_labels(self):
        """return the number of labeles in this dataset"""
        if not self._n_labels:
            self.set_labels_attr()

        return self._n_labels

    @property
    def labels(self):
        """return the list of labels in this dataset"""
        if not self._labels:
            self.set_labels_attr()

        return self._labels
