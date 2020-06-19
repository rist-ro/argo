"""
Module for managing dSprites dataset, mainly used for learning disentangled representations
"""

import numpy as np

import os.path
import urllib.request

from datasets.Dataset import TRAIN, VALIDATION, TEST
from .ImageDataset import ImageDataset
from scipy.io import loadmat

import pdb


class DSprites(ImageDataset):
    """
    This class manages the dSprites dataset, properties of the datasets are uniquely determined
    by the params dictionary. The dataset is binary, with the option of converting it from {0, 1} to {-1, 1}

    It compares the parameters and complete them with the default one. It then return a unique id identifier

    Parameters
    ---------

    params : dict
        dictionary that can contain
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | params Key  | values    | default v |  id short | Short description                                           |
         +=============+===========+===========+===========+=============================================================+
         | data_dir    | path str  | None      |           | path of the dataset                                         |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | subsampilng |  integer  | None      |"-subsamp" | Reduce the dataset providing 1 data over `subsamping`       |
         |             |           |           |           | samples                                                     |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | clip_low    | bool      | None      | "-clipL"  | clip the dataset to a minimum value (used to avoid zero     |
         |             |           |           |           | gradient)    (-clipLH in case of also high)                 |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | clip_high   | bool      | None      | "-clipH"  | clip the dataset to a max value                             |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | id_note     | string    | ""        | id_note   | Arbitrary string to append to the id                        |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | test_proportion | float | 0.1      |            | proportion of data to include in validation set             |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | validation_proportion | float | 0.1 |           | proportion of data to include in test set                   |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | normalize     | bool    | True      |           | flag to indicate if you want to covert data to {-1, 1}      |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+

    """

    default_params = {
        'stochastic': 0,
        'subsampling': None,
        'clip_high': None,
        'clip_low': None,
        'id_note': None,
        'test_proportion': 0.1,
        'validation_proportion': 0.1,
        'normalize': True,
    }

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._normalize_data = self._params['normalize']

        self.data_dir = "/ssd_data/datasets/dSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

        self.img_rows = 64
        self.img_cols = 64
        self.factors_frequencies = [3, 6, 40, 32, 32]
        self.num_factors = len(self.factors_frequencies)

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_data()

        # choose a subset
        if self._params['subsampling']:
            self._train_set_x, self._train_set_y = \
                self.sub_sample(self._train_set_x, self._train_set_y, self._params['subsampling'])
            self._test_set_x, self._test_set_y = \
                self.sub_sample(self._test_set_x, self._test_set_y, self._params['subsampling'])


        # clip
        clip_low = self._params['clip_low']
        clip_high = self._params['clip_high']
        if (clip_low is not None) or (clip_high is not None):
            m = clip_low if clip_low is not None else 0
            M = clip_high if clip_high is not None else 1
            self._train_set_x = np.clip(self._train_set_x, a_min=m, a_max=M)
            self._test_set_x = np.clip(self._test_set_x, a_min=m, a_max=M)

    implemented_params_keys = ['dataName', 'subsampling', 'clip_high', 'clip_low',
                               'data_dir', 'id_note', 'stochastic', 'test_proportion', 'validation_proportion',
                               'normalize']  # all the admitted keys

    def load_data(self):
        dtype = np.float32
        dtype_labels = np.int64

        dataset_zip = np.load(self.data_dir, encoding='bytes')
        imgs = dataset_zip['imgs']
        imgs = imgs[:, :, :, None]  # make into 4d tensor

        # shuffle data
        n_data = len(imgs)
        labels = dataset_zip['latents_classes']

        randomized_dataset_index = np.random.permutation(n_data)

        imgs = imgs[randomized_dataset_index]
        labels = labels[randomized_dataset_index]

        # convert to float32
        imgs = imgs.astype(dtype)

        if self._normalize_data:
            # convert to -1, 1
            imgs = imgs * 2 - 1

        #split in train-validation-test
        validation_split = (1 - (self._params["test_proportion"] + self._params["validation_proportion"])) * n_data
        test_split = validation_split + self._params["test_proportion"] * n_data

        validation_split, test_split = int(validation_split), int(test_split)

        train_set_x, validation_set_x, test_set_x = imgs[:validation_split, :], \
                                                    imgs[validation_split: test_split, :], \
                                                    imgs[test_split:, :]
        train_set_y, validation_set_y, test_set_y = labels[:validation_split, :], \
                                                    labels[validation_split: test_split, :], \
                                                    labels[test_split:, :]

        return train_set_x, train_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y


    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?
        DSprites.check_params_impl(params)

        id = 'DSprites'

        # subsampling
        if params['subsampling']:
            id += '-ss' + str(params['subsampling'])

        # clip
        # TODO The parameters of clip should be the values to which you clip
        clip_high = False
        if params['clip_high']:
            id += '-cH'
            clip_high = True

        if params['clip_low']:
            id += '-cL'
            if clip_high:
                id += "H"

        # id note (keep last)
        if params['id_note']:
            id += params['id_note']

        return id

    @staticmethod
    def sub_sample(data_set_x, data_set_y, subsampling):
        """
        return a value every "subsampling"

        :param data_set_x
        :param data_set_y
        :param subsampling: integer < dim(data_set)
        :return: dataset_x, dataset_y
        """

        len_train = len(data_set_x)
        reshuf_index_train = np.random.permutation(len_train)
        new_len_train = int(len_train / subsampling)

        data_set_x = data_set_x[reshuf_index_train[:new_len_train]]
        data_set_y = data_set_y[reshuf_index_train[:new_len_train]]

        return data_set_x, data_set_y

    @staticmethod
    def class_filter(data_set_x, data_set_y, classes, position_label):
        """
        return the dataset with labels in the list classes

        :param data_set_x: data
        :param data_set_y: labels
        :param classes:    list of classes
        :param position_label:  list of classes
        :return: (dataset_x, dataset_y) with filtered elemnts not in classes
        """

        ix_mtch_class_train = np.in1d(data_set_y, classes)
        data_set_x = data_set_x[ix_mtch_class_train]
        data_set_y = data_set_y[ix_mtch_class_train]
        if position_label:

            def replace_with_position(label_set, classes):
                label_set_new = np.copy(label_set)
                for ix, class_ in enumerate(classes): label_set_new[label_set == class_] = ix
                return label_set_new

            data_set_y = replace_with_position(data_set_y, classes)

        return data_set_x, data_set_y

    @property
    def input_size(self):
        return self.img_rows * self.img_cols

    @property
    def output_size(self):
        pass

    @property
    def color_images(self):
        return 0

    @property
    def image_shape(self):
        return self.img_rows, self.img_cols, 1
