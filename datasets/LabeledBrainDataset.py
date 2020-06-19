"""
Module for managing labeled brain datasets
"""
from datasets.BrainDataset import BrainDataset, modalities, NPROCS
import numpy as np
import os
import fnmatch
import tensorflow as tf
import re

import PIL
import json
import pdb


class LabeledBrainDataset(BrainDataset):
    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)
        self._data_dir = self._params['data_dir']
        self._labels_file = self._params['labels_file']
        self._split_file = self._params['split_file']

        if 'slices' in self._params.keys():
            self._slices = self._params['slices']
        else:
            self._slices = None
        # options for each dataset

        self._no_of_classes = self._params['no_of_classes']

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_float_brains(self._data_dir)

    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        id = ''

        id += super().dataset_id(params)
        return id

    def load_float_brains(self, data_dir):
        datasets_tuple = self.load_file_names(data_dir, 'train')
        datasets_tuple_validation = self.load_file_names(data_dir, 'validation')
        datasets_tuple_test = self.load_file_names(data_dir, 'test')
        # pdb.set_trace()

        print('---------DATASET TUPLE------------', datasets_tuple.shape)
        train_set_x, train_set_y = datasets_tuple

        print('---------DATASET TUPLE VALIDATION------------', datasets_tuple_validation.shape)
        validation_set_x, validation_set_y = datasets_tuple_validation

        print('---------DATASET TUPLE TEST------------', datasets_tuple_test.shape)
        test_set_x, test_set_y = datasets_tuple_test

        print('--------------X SHAPE-----------------')
        channels_no = len(self._modalities) if self._modalities != None else 1
        self._train_set_x_shape = np.load(datasets_tuple[0, 0]).shape + (channels_no,)
        # self._train_set_y_shape = self.get_label(datasets_tuple[1, 0]).shape
        # self._train_set_y_shape = (4,)
        # self._train_set_y_shape = (1,)
        if self._resize is not None:
            self._train_set_x_shape = (self._resize, self._resize, channels_no)
        print(self._train_set_x_shape)

        return train_set_x, train_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y

    # overriding
    def load_file_names(self, root, data_type):
        original_files = []
        label_files = []
        with open(self._split_file, 'r') as file:
            files_to_find = json.load(file)[data_type]
            for path, dirs, files in os.walk(root):
                if self._modalities is not None:
                    reg_filter = '*_' + str(modalities[self._modalities[0]]) + '_*'
                    for f in fnmatch.filter(files, reg_filter):
                        idx = f.find('_' + str(modalities[self._modalities[0]]))
                        # idx = f.find('_')
                        label_file_name = f[:idx]
                        if label_file_name in files_to_find:
                            fullname = root + '/' + f
                            if self._slices is not None:
                                slice = re.findall('_([0-9][0-9]*)', f)
                                if self._slices[0] <= int(slice[0]) <= self._slices[1]:
                                    original_files.append(fullname)
                                    label_files.append(label_file_name)
                            else:
                                original_files.append(fullname)
                                label_files.append(label_file_name)
                else:
                    for f in files:
                        idx = f.find('_')
                        label_file_name = f[:idx]
                        if label_file_name in files_to_find:
                            fullname = root + '/' + f
                            # idx = f.find('_' + str(modalities['T2']))
                            original_files.append(fullname)
                            label_files.append(label_file_name)
        dataset_tuple = [original_files, label_files]
        return np.asarray(dataset_tuple)

    def get_label(self, filename):
        # label = -1
        with open(self._labels_file, 'r') as json_file:
            labels_dict = json.load(json_file)
            # pdb.set_trace()
            # if filename in labels_dict:
            label = np.nonzero(labels_dict[filename])[0].astype(np.int32)[0]
            # if label == 3:
            #     label = np.array([2], dtype=np.int32)[0]
            # return np.nonzero(labels_dict[filename])[0].astype(np.int32)[0]
            return label
            # return np.asarray(labels_dict[filename], dtype=np.int32)
            # return float(np.nonzero(labels_dict[filename])[0][0])

    # return label

    # overriding
    def dataset_map(self, dataset, datasets_tuple):
        output_types = self.get_output_types(datasets_tuple)
        output_shapes = self.get_output_shapes(datasets_tuple)

        def load_function(n):
            filename = full_data[n][0]
            label_filename = full_data[n][1]
            result = np.empty(output_shapes[0], np.float32)
            if self._modalities != None:
                for i, modality in enumerate(self._modalities):
                    modality_filename = str.replace(str(filename), modalities[self._modalities[0]],
                                                    modalities[modality])
                    image = self.load_slice_from_file(str(modality_filename))
                    if self._resize is not None:
                        image = np.array(
                            PIL.Image.fromarray(image).resize([self._resize, self._resize]))
                    result[:, :, i] = image
            else:
                # image = self.load_slice_from_file(self._data_dir + '/' + str(filename))
                image = self.load_slice_from_file(str(filename))
                if self._resize is not None:
                    image = np.array(
                        PIL.Image.fromarray(image).resize([self._resize, self._resize]))
                result = image.reshape([image.shape[0], image.shape[1], 1])
            label = self.get_label(label_filename)
            # label = np.reshape(label, [label.shape[0], label.shape[1], 1])
            return result, label
            # return result

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
        # label = self.get_label(datasets_tuple[1][0]).astype(np.float32)
        # label = float(self.get_label(datasets_tuple[1][0]))
        output_shapes = tuple([image.shape + (channels_no,), ()])
        if self._resize is not None:
            output_shapes = ((self._resize, self._resize, channels_no), ())

        return output_shapes

    # overriding
    def get_output_types(self, datasets_tuple):
        image = np.load(datasets_tuple[0][0]).astype(np.float32)
        output_types = tuple([tf.as_dtype(image.dtype), tf.int32])
        # label = self.get_label(datasets_tuple[1][0]).astype(np.float32)
        # label = float(self.get_label(datasets_tuple[1][0]))
        # output_types = tuple([tf.as_dtype(image.dtype), tf.as_dtype(label.dtype)])

        return output_types

    def label_to_name(self, label):
        label_to_name_M_dict = {0: "22-25",
                                1: "26-30",
                                2: "31-35",
                                3: "36+"}
        return label_to_name_M_dict[label]

    # @property
    # def output_size(self):
    #     return 4

    @property
    def x_shape(self):
        """return the shape of an input sample for the train loop"""
        return self.x_shape_train

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

    # overriding
    @property
    def y_shape(self):
        """return the shape of an output sample"""
        train_set_y = getattr(self, 'train_set_y', None)
        if train_set_y is None:
            raise ValueError("this dataset does not have y set")
        # return self._train_set_y_shape
        return ()

    # overriding
    @property
    def n_labels(self):
        """return the number of labeles in this dataset"""
        if not self._n_labels:
            self._n_labels = self._no_of_classes

        return self._n_labels

    @property
    def labels(self):
        """return the list of labels in this dataset"""
        if not self._labels:
            self._labels = np.asarray([0, 1, 2], dtype=np.int32)

        return self._labels
