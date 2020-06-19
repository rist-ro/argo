"""
Module for managing Brain datasets
"""
from datasets.ImageDataset import ImageDataset
import numpy as np
import os
import fnmatch
import tensorflow as tf

import PIL

NPROCS = 40

modalities = {"FLAIR": 'flair',
              "T1": 't1',
              "T1c": 't1ce',
              "T2": 't2',
              "SEG": 'seg'}


class BrainDataset(ImageDataset):
    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._caching_bool = False
        self._shuffling_cache = None

        default_data_dir = ''
        self._data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir

        if 'resize' in self._params.keys():
            self._resize = self._params['resize']
        else:
            self._resize = None

        if 'modalities' in self._params.keys():
            self._modalities = self._params['modalities']

            if self._modalities[0] == 'SEG':
                self.use_mask = True
        else:
            self._modalities = None

    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        BrainDataset.check_params_impl(params)

        id = ''
        if 'options' in params.keys():
            id += '-' + params['options']

        if 'modalities' in params.keys():
            id += '-' + '_'.join(params['modalities'])

        if 'resize' in params.keys():
            id += '-' + str(params['resize'])

        return id

    def load_float_brains(self, data_dir):
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
        channels_no = len(self._modalities) if self._modalities != None else 1
        self._train_set_x_shape = np.load(data_dir + '/' + datasets_tuple[0]).shape + (channels_no,)
        print(self._train_set_x_shape)

        return train_set_x, validation_set_x, test_set_x

    def load_file_names(self, root, data_type):
        file_names = []
        for path, dirs, files in os.walk(root + '/' + data_type):
            if  self._modalities != None:
                reg_filter = '*_' + str(modalities[self._modalities[0]]) + '_*'
                for f in fnmatch.filter(files, reg_filter):
                    file_names.append(data_type + '/' + f)
            else:
                for f in files:
                    file_names.append(data_type + '/' + f)
        file_names = np.asarray(file_names)
        return file_names

    def load_slice_from_file(self, file):
        slice = np.load(file)
        slice = slice.astype(np.float32)
        slice = np.asarray(slice)
        return slice

    def load_slices_from_file(self, file):
        slices = np.load(file)
        slices = slices.astype(np.float32)
        slices = np.asarray(slices)
        return slices

    # overriding
    def dataset_map(self, dataset, datasets_tuple):
        output_types = self.get_output_types(datasets_tuple)
        output_shapes = self.get_output_shapes(datasets_tuple)

        def load_function(n):
            filename = full_data[n][0]
            result = np.empty(output_shapes[0], np.float32)
            if self._modalities is not None:
                for i, modality in enumerate(self._modalities):
                    modality_filename = str.replace(str(filename), modalities[self._modalities[0]],
                                                    modalities[modality])
                    image = self.load_slice_from_file(self._data_dir + '/' + str(modality_filename))
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
            return result
        
        def apply_py_function(n):
            return tuple(tf.py_func(load_function, [n], output_types))
        
        full_data = list(zip(*datasets_tuple))

        dataset = dataset.map(apply_py_function, num_parallel_calls=NPROCS)
        #dataset = dataset.map(
        #    lambda n: tuple(tf.py_func(load_function,
        #                               [n], output_types)
        #                    ), num_parallel_calls=NPROCS)
        
        return dataset

    def get_output_shapes(self, datasets_tuple):
        channels_no = len(self._modalities) if self._modalities is not None else 1

        # if we resize, there's no need to load an image and get its shape
        if self._resize is not None:
            output_shapes = ((self._resize, self._resize, channels_no),)
        else:
            image = np.load(self._data_dir + '/' + datasets_tuple[0][0]).astype(np.float32)
            output_shapes = tuple([image.shape + (channels_no,)])

        return output_shapes

    def get_output_types(self, datasets_tuple):
        image = np.load(self._data_dir + '/' + datasets_tuple[0][0]).astype(np.float32)
        output_types = tuple([tf.as_dtype(image.dtype)])

        return output_types

    def get_raw_elements(self, dataset_str, index_list=None):
        attribute_name = dataset_str + "_set_x"
        images = []
        if index_list is not None:
            for file in getattr(self, attribute_name)[index_list]:
                image = self.load_slices_from_file(self._data_dir + '/' + str(file))
                images.append(image)
            images = np.asarray(images)
            images = images.reshape([images.shape[0], images.shape[1], images.shape[2], 1])
            x_data = images
        else:
            x_data = getattr(self, attribute_name)
        return x_data

