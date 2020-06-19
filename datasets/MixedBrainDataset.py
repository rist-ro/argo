"""
Module for managing multiple brain datasets at once
"""
from datasets.BrainDataset import BrainDataset, modalities, NPROCS
import numpy as np
import os
import fnmatch
import tensorflow as tf

import PIL
import pdb


class MixedBrainDataset(BrainDataset):
    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)
        self._data_dirs = self._params['data_dirs']
        self._training_data_proportion = self._params['training_data_proportion'] \
            if 'training_data_proportion' in params else [1.0 for _ in range(len(self._data_dirs))]

        # options for each dataset

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_float_brains(self._data_dirs, self._training_data_proportion)

    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        id = 'MixedBrainDataset'

        id += super().dataset_id(params)
        if 'training_data_proportion' in params.keys():
            id += "-p" + "_".join([str(val) for val in params['training_data_proportion']])
        return id

    def load_float_brains(self, data_dirs, proportions):
        datasets_tuple = []
        datasets_tuple_validation = []
        datasets_tuple_test = []
        for data_dir, proportion in zip(data_dirs, proportions):
            datasets_tuple = np.append(datasets_tuple, self.load_file_names(data_dir, 'train',
                                                                            proportion=proportion))
            datasets_tuple_validation = np.append(datasets_tuple_validation,
                                                  self.load_file_names(data_dir, 'validation'))
            datasets_tuple_test = np.append(datasets_tuple_test, self.load_file_names(data_dir, 'test'))

        datasets_tuple = np.tile(datasets_tuple, (2, 1))
        datasets_tuple_validation = np.tile(datasets_tuple_validation, (2, 1))
        datasets_tuple_test = np.tile(datasets_tuple_test, (2, 1))
        datasets_tuple = np.asarray(datasets_tuple)
        datasets_tuple_validation = np.asarray(datasets_tuple_validation)
        datasets_tuple_test = np.asarray(datasets_tuple_test)

        print('---------DATASET TUPLE------------', datasets_tuple.shape)
        train_set_x, train_set_y = datasets_tuple

        print('---------DATASET TUPLE VALIDATION------------', datasets_tuple_validation.shape)
        validation_set_x, validation_set_y = datasets_tuple_validation

        print('---------DATASET TUPLE TEST------------', datasets_tuple_test.shape)
        test_set_x, test_set_y = datasets_tuple_test

        print('--------------X SHAPE-----------------')
        channels_no = len(self._modalities) if self._modalities != None else 1
        self._train_set_x_shape = np.load(datasets_tuple[0, 0]).shape + (channels_no,)
        if self._resize is not None:
            self._train_set_x_shape = (self._resize, self._resize, channels_no)
        print(self._train_set_x_shape)

        return train_set_x, train_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y

    # overriding
    def load_file_names(self, root, data_type, proportion=1.0):
        file_names = []
        for path, dirs, files in os.walk(root + '/' + data_type):
            if self._modalities is not None:
                reg_filter = '*_' + str(modalities[self._modalities[0]]) + '_*'
                for f in fnmatch.filter(files, reg_filter):
                    file_names.append(root + '/' + data_type + '/' + f)
            else:
                for f in files:
                    file_names.append(root + '/' + data_type + '/' + f)
        file_names = np.asarray(file_names)
        file_names = file_names[:int(proportion * len(file_names))]
        return file_names

    def get_label(self, filename):
        if 'HCP' in filename:
            return 0
        if 'BRATS' in filename:
            return 1
        # todo implement when the tumour mask is given and the label is [1,1]

    # overriding
    def dataset_map(self, dataset, datasets_tuple):
        output_types = self.get_output_types(datasets_tuple)
        output_shapes = self.get_output_shapes(datasets_tuple)

        def load_function(n):
            filename = full_data[n][0]
            result = np.empty(output_shapes[0], np.float32)
            modality_filename = filename
            if self._modalities is not None:
                for i, modality in enumerate(self._modalities):
                    modality_filename = str.replace(str(filename), modalities[self._modalities[0]],
                                                    modalities[modality])
                    image = self.load_slice_from_file(str(modality_filename))
                    if self._resize is not None:
                        image = np.array(
                            PIL.Image.fromarray(image).resize([self._resize, self._resize]))
                    result[:, :, i] = image
            else:
                image = self.load_slice_from_file(self._data_dir + '/' + str(filename))
                if self._resize is not None:
                    image = np.array(
                        PIL.Image.fromarray(image).resize([self._resize, self._resize]))
                result = image.reshape([image.shape[0], image.shape[1], 1])
            label = self.get_label(modality_filename)
            return result, np.int32(label)

        full_data = list(zip(*datasets_tuple))

        dataset = dataset.map(
            lambda n: tuple(tf.py_func(load_function,
                                       [n], output_types)
                            ), num_parallel_calls=NPROCS)
        return dataset

    # overriding
    def get_output_shapes(self, datasets_tuple):
        image = np.load(datasets_tuple[0][0]).astype(np.float32)
        channels_no = len(self._modalities) if self._modalities is not None else 1
        output_shapes = tuple([image.shape + (channels_no,), ()])
        if self._resize is not None:
            output_shapes = ((self._resize, self._resize, channels_no), ())

        return output_shapes

    # overriding
    def get_output_types(self, datasets_tuple):
        image = np.load(datasets_tuple[0][0]).astype(np.float32)
        output_types = tuple([tf.as_dtype(image.dtype), tf.int32])

        return output_types

    # overriding
    @property
    def x_shape_train(self):
        return self._train_set_x_shape

    # overriding
    # @property
    # def y_shape_train(self):
    #     return self._train_set_y_shape

    # overriding
    @property
    def x_shape_eval(self):
        return self._train_set_x_shape

    @property
    def n_labels(self):
        """return the number of labeles in this dataset"""
        return 2

    @property
    def data_dirs(self):
        return self._data_dirs
