"""
Module for managing BRATS dataset
"""
from datasets.BrainDataset import modalities
from datasets.LabeledBrainDataset import LabeledBrainDataset

import os
import fnmatch
import numpy as np
import re
import json
import pdb

NPROCS = 40

TRAIN_LOOP = "train_loop"
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


class BRATSLabeled(LabeledBrainDataset):
    def __init__(self, params):
        super().__init__(params)

        self._no_of_classes = 4

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_float_brains(self._data_dir)

    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        id = 'BRATSLabeled'
        id += super().dataset_id(params)

        return id

    # overriding
    @property
    def x_shape_train(self):
        return self._train_set_x_shape

    # overriding
    @property
    def x_shape_eval(self):
        return self._train_set_x_shape

    # overriding
    def get_label(self, filename):
        # label = -1
        with open(self._labels_file, 'r') as json_file:
            labels_dict = json.load(json_file)
            label = np.nonzero(labels_dict[filename])[0].astype(np.int32)[0]
            return label

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
                        # idx = f.find('_' + str(modalities[self._modalities[0]]))
                        # idx = f.find('_')
                        # label_file_name = f[:idx]
                        start_idx = f.find('Brats')
                        end_idx = f.find('_' + str(modalities[self._modalities[0]]))
                        label_file_name = f[start_idx:end_idx]
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
        # pdb.set_trace()
        dataset_tuple = [original_files, label_files]
        return np.asarray(dataset_tuple)

    # def load_file_names(self, root, data_type):
    #     original_files = []
    #     label_files = []
    #     for path, dirs, files in os.walk(root + '/' + data_type):
    #         if self._modalities != None:
    #             reg_filter = '*_' + str(modalities[self._modalities[0]]) + '_*'
    #             for f in fnmatch.filter(files, reg_filter):
    #                 fullname = root + '/' + data_type + '/' + f
    #                 start_idx = f.find('Brats')
    #                 end_idx = f.find('_' + str(modalities[self._modalities[0]]))
    #                 label_file_name = f[start_idx:end_idx]
    #                 original_files.append(fullname)
    #                 label_files.append(label_file_name)
    #         else:
    #             for f in files:
    #                 fullname = root + '/' + data_type + '/' + f
    #                 start_idx = f.find('BRATS')
    #                 end_idx = f.find('_' + str(modalities['T2']))
    #                 label_file_name = f[start_idx:end_idx]
    #                 original_files.append(fullname)
    #                 label_files.append(label_file_name)
    #     dataset_tuple = [original_files, label_files]
    #     return np.asarray(dataset_tuple)
