"""
Module for managing BRATS dataset
"""
from datasets.BrainDataset import modalities
from datasets.LabeledBrainDataset import LabeledBrainDataset

import os
import fnmatch
import numpy as np
import pdb

NPROCS = 40

TRAIN_LOOP = "train_loop"
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


class HCPLabeled(LabeledBrainDataset):
    def __init__(self, params):
        super().__init__(params)

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_float_brains(self._data_dir)

    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        id = 'HCPLabeled'
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

