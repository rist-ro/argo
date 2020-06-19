"""
Module for managing BRATS dataset
"""
from datasets.ImageDataset import ImageDataset
import numpy as np
import os
import fnmatch
import tensorflow as tf
import pdb

NPROCS = 40

WIDTH = 240
HEIGHT = 240

TRAIN_LOOP = "train_loop"
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

machines = {"FLAIR": 0,
            "T1": 1,
            "T1c": 2,
            "T2": 3}

import PIL


class BRATScnnLazyLoading_perSlice_all(ImageDataset):
    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        default_data_dir = '/ssd_data/BRATS_data/all_slices_separately_one'
        self._data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir
        # self._data_dir = default_data_dir

        self._caching_bool = False
        self._shuffling_cache = None
        self._machine = self._params['machine']

        if 'resize' in self._params.keys():
            self._resize = self._params['resize']
        else:
            self._resize = None


        self._train_set_x, self._validation_set_x, self._test_set_x = self.load_float_brats(self._data_dir)

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        BRATScnnLazyLoading_perSlice_all.check_params_impl(params)

        id = 'BRATScnnLazyLoading_perSlice'
        id += '-' + params['options']
        if params['machine']:
            id += '-' + params['machine']

        if 'resize' in params.keys():
            id += '-' + str(params['resize'])

        return id

    def load_float_brats(self, data_dir):
        # data_dir = '/ssd_data/BRATS_data/all_slices_separately_one'

        datasets_tuple = self.load_file_names(data_dir, 'train')
        print('---------DATASET TUPLE------------', datasets_tuple.shape)

        train_set_x = datasets_tuple

        datasets_tuple_validation = self.load_file_names(data_dir, 'validation')
        print('---------DATASET TUPLE VALIDATION------------', datasets_tuple_validation.shape)

        validation_set_x = datasets_tuple_validation

        datasets_tuple_test = self.load_file_names(data_dir, 'test')
        print('---------DATASET TUPLE TEST------------', datasets_tuple_test.shape)

        test_set_x = datasets_tuple_test

        print('--------------X SHAPE-----------------')
        self._train_set_x_shape = np.load(self._data_dir + '/' + datasets_tuple[0]).shape + (1,)
        print(self._train_set_x_shape)

        return train_set_x, validation_set_x, test_set_x

    def load_slices_from_files(self, root):
        slices = []
        for path, dirs, files in os.walk(root):
            for f in files:
                fullname = os.path.abspath(os.path.join(path, f))
                slice = np.load(fullname)
                slice = slice.astype(np.float32)
                slice = slice / (slice.max())
                slices.append(slice)
        slices = np.asarray(slices)

        return slices

    # def load_file_names(self, root):
    #     file_names = []
    #     for path, dirs, files in os.walk(root):
    #         for f in files:
    #             file_names.append(f)
    #     file_names = np.asarray(file_names)
    #     return file_names

    def load_file_names(self, root, data_type):
        file_names = []
        for path, dirs, files in os.walk(root + '/' + data_type):
            if self._machine != '':
                reg_filter = '*_' + str(machines[self._machine]) + '_*'
                for f in fnmatch.filter(files, reg_filter):
                    file_names.append(data_type + '/' + f)
            else:
                for f in files:
                    file_names.append(data_type + '/' + f)
        file_names = np.asarray(file_names)
        return file_names

    def load_map_filename_slice(self, root):
        map_fn_s = []
        for path, dirs, files in os.walk(root):
            for f in files:
                for slice in range(0, 130):
                    map_fn_s.append((f, slice))
        map_fn_s = np.asarray(map_fn_s)
        return map_fn_s

    def load_slices_from_file(self, file):
        slices = np.load(file)
        slices = slices.astype(np.float32)
        slices = slices / (slices.max())
        slices = np.asarray(slices)
        return slices

    def load_slice_from_file(self, file):
        slice = np.load(file)
        slice = slice.astype(np.float32)
        slice = np.asarray(slice)
        return slice

    # overriding
    def dataset_map(self, dataset, datasets_tuple):
        output_types = self.get_output_types(datasets_tuple)

        def load_function(n):
            filename = full_data[n][0]
            image = self.load_slice_from_file(self._data_dir + '/' + str(filename))
            if self._resize is not None:
                image = np.array(
                    PIL.Image.fromarray(image).resize([self._resize, self._resize]))
            reshaped_img = image.reshape([image.shape[0], image.shape[1], 1])
            return reshaped_img

        full_data = list(zip(*datasets_tuple))

        dataset = dataset.map(
            lambda n: tuple(tf.py_func(load_function,
                                       [n], output_types)
                            ), num_parallel_calls=NPROCS)
        return dataset

    def get_output_shapes(self, datasets_tuple):
        image = np.load(self._data_dir + '/' + datasets_tuple[0][0]).astype(np.float32)
        output_shapes = tuple([image.shape + (1,)])
        if self._resize is not None:
            output_shapes = ((self._resize, self._resize, 1),)

        return output_shapes

    def get_output_types(self, datasets_tuple):
        image = np.load(self._data_dir + '/' + datasets_tuple[0][0]).astype(np.float32)
        output_types = tuple([tf.as_dtype(image.dtype)])

        return output_types

    def get_raw_elements(self, dataset_str, index_list=None):
        attribute_name = dataset_str + "_set_x"
        # print('-----------ATTRIBUTE NAME-----------', attribute_name)
        # print(getattr(self, attribute_name)[index_list])
        # pdb.set_trace()
        images = []
        if index_list is not None:
            for file in getattr(self, attribute_name)[index_list]:
                image = self.load_slices_from_file(self._data_dir + '/' + str(file))
                # print(image.shape)
                images.append(image)
            images = np.asarray(images)
            images = images.reshape([images.shape[0], images.shape[1], images.shape[2], 1])
            x_data = images
        else:
            x_data = getattr(self, attribute_name)
        return x_data

    # LUIGI: What is the need of copy and paste a method from the parent class in the child
    # without making any modification? Talk to me when you read this    
    '''
    def get_dataset_iterator(self, batch_size, dataset_str, shuffle, repeat, augment):
        is_perturbed = False
        datasets_tuple = None

        # create Dataset objects using the data previously downloaded
        if dataset_str == TRAIN:
            datasets_tuple = (self.train_set_x,)
            if hasattr(self, "perturbed_train_set_x") and self.perturbed_train_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_train_set_x,)
                is_perturbed = True
            if self._train_set_y is not None:
                datasets_tuple = datasets_tuple + (self.train_set_y,)

        elif dataset_str == VALIDATION:
            datasets_tuple = (self.validation_set_x,)
            if hasattr(self, "perturbed_validation_set_x") and self.perturbed_validation_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_validation_set_x,)
                is_perturbed = True
            if self._validation_set_y is not None:
                datasets_tuple = datasets_tuple + (self.validation_set_y,)

        elif dataset_str == TEST:
            datasets_tuple = (self.test_set_x,)
            if hasattr(self, "perturbed_test_set_x") and self.perturbed_test_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_test_set_x,)
                is_perturbed = True
            if self._test_set_y is not None:
                datasets_tuple = datasets_tuple + (self.test_set_y,)

        else:
            raise Exception("dataset not recognized (accepted values are: train, validation and test)")

        # # CREATE TF DATASET from slices
        # # from_tensor_slices is storing dataset in the graph thus making checkpoints huge
        # dataset = tf.data.Dataset.from_tensor_slices(datasets_tuple)
        # # CREATE TF DATASET from slices

        # # CREATE TF DATASET from generator
        # def generator():
        #     for sample in zip(*datasets_tuple):
        #         yield sample
        #
        # output_types = tuple([tf.as_dtype(ds[0].dtype) for ds in datasets_tuple])
        # output_shapes = tuple([ds[0].shape for ds in datasets_tuple])
        #
        # dataset = tf.data.Dataset.from_generator(generator, output_types=output_types,
        #                                             output_shapes=output_shapes)
        # # CREATE TF DATASET from generator
        #

        # CREATE TF DATASET with map and py_func

        n_samples = datasets_tuple[0].shape[0]
        print('------------MY N SAMPLES------------', n_samples)
        dataset = tf.data.Dataset.range(n_samples)

        output_shapes = self.get_output_shapes(datasets_tuple)
        print('--------------output_shapes----------------')
        print(output_shapes)

        # why should anything be None in datasets_tuple? it is not clear that it would work with the oytput_shapes...
        # output_types = tuple([tf.as_dtype(ds[0].dtype) for ds in datasets_tuple if ds is not None])
        # output_shapes = tuple([ds[0].shape for ds in datasets_tuple if ds is not None])

        dataset = self.dataset_map(dataset, datasets_tuple)

        def _set_shapes(*nodes):
            for node, outshape in zip(nodes, output_shapes):
                node.set_shape(outshape)
            return nodes

        dataset = dataset.map(_set_shapes, num_parallel_calls=NPROCS)
        # CREATE TF DATASET with map and py_func

        # caching before shuffling and batching for super cow speed
        if self._caching_bool:
            dataset = dataset.cache()

        # PREPROCESS DATA (AUGMENT IF NEEDED)
        # if augmentation_bool:
        #     dataset = dataset.map(partial(self._preprocess_element, is_perturbed), num_parallel_calls=NPROCS)

        # SHUFFLE, REPEAT and BATCH
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
        return iterator, is_perturbed
    '''

    # overriding
    @property
    def x_shape_train(self):
        return self._train_set_x_shape

    # overriding
    @property
    def x_shape_eval(self):
        return self._train_set_x_shape
