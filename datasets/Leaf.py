"""
Module for managing the IRIS dataset
"""

import numpy as np
import pandas as pd
import pdb

from .Dataset import Dataset


class Leaf(Dataset):
    """
    This class manage the dataset Leaf, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    https://www.kaggle.com/c/leaf-classification/data
    The images have been resized so they are the same size
    """

    default_params = {

        'test_and_validation': 150,
        'data_dir':            "/ssd_data/datasets/leaf"
    }

    classification = False  # true if

    implemented_params_keys = ['dataName']  # all the admitted keys

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._data_dir = self._params['data_dir']

        # get info from the metadata
        self.labels_df = self.read_labels()

        self._binary_input = True

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_data()

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        Leaf.check_params_impl(params)

        id = 'Leaf'
        return id

    def read_labels(self):
        """
        :return:
        """
        # load csv file
        train_csv_filename = self._data_dir + '/train.csv'
        test_csv_filename = self._data_dir + '/test.csv'
        train_label_df = pd.read_csv(train_csv_filename)
        test_label_df = pd.read_csv(test_csv_filename)

        label_df = {
            "train": train_label_df,
            "test":  test_label_df}
        return label_df

    def load_data(self):
        # Original size of data is huuuge
        # fileName = self._data_dir + '/images.npy'
        fileName = self._data_dir + '/images_small.npy'
        order_fileName = self._data_dir + '/order_images.npy'

        order = np.load(order_fileName).astype(np.int32) - 1
        X = np.load(fileName).astype(np.float32) / 255
        X = np.take(X, order, axis=0)
        n_files = X.shape[0]

        # train
        train_Y = np.asarray(self.labels_df["train"]["id"] - 1)

        train_X = np.take(X, train_Y, axis=0)
        train_files = train_X.shape[0]

        randomized_dataset_index = np.random.permutation(train_files)
        train_X = np.expand_dims(train_X[randomized_dataset_index],axis=-1)
        train_Y = train_Y[randomized_dataset_index]

        # test
        test_Y = np.asarray(self.labels_df["test"]["id"] - 1)

        test_X = np.take(X, test_Y, axis=0)
        test_files = test_X.shape[0]

        randomized_dataset_index = np.random.permutation(test_files)
        test_X = np.expand_dims(test_X[randomized_dataset_index],axis=-1)
        test_Y = test_Y[randomized_dataset_index]

        val_split = train_files - self._params['test_and_validation']

        train_set_x, validation_set_x, test_set_x = train_X[:val_split], train_X[val_split:], test_X
        # train_set_y, validation_set_y, test_set_y = train_Y[:val_split], train_Y[val_split:], test_Y
        train_set_y, validation_set_y, test_set_y = None, None, None

        return train_set_x, train_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y

    @property
    def color_images(self):
        return 0

    @property
    def image_shape(self):
        return (32, 50, 1)  # the last number is the channel
