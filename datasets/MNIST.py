"""
Module for managing MNIST dataset
"""

import numpy as np

import tensorflow as tf

import os

import gzip
import pickle

from .Dataset import NPROCS

import pdb

from .ImageDataset import ImageDataset

from .utils import normalize, min_max_data_np


class MNIST(ImageDataset):
    """
    This class manage the dataset MNIST, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    Parameters
    ---------

    params : dict
        dictionary that can contain
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | params Key  | values    | default v |  id short | Short description                                           |
         +=============+===========+===========+===========+=============================================================+
         |   binary    |  -1,1      |  0        | "-c", "-d"| load continuous or binary MNIST                             |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         |  stochastic |  -1,1      |  0        |   "-st"   | sample using Bernoulli from the continuous MNIST after every|
         |             |           |           |           | epoch during training, see IWAE and LVAE papers that claim  |
         |             |           |           |           | this techniques reduces overfitting; this function only     |
         |             |           |           |           | loads continuous MNIST to be used later                     |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | data_dir    | path str  | None      |           | path of the dataset. In some cases cannot be set            |
         |             |           |           |           | (non binary mnist only)                                     |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | classes     | e.g.      |           |           | select a touple of subclasses of digits, e.g. [0,1]           |
         |             | (3,..,9)  |  All      | "-cl"+val |                                                             |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | position_label | bool   | True      |if false   | labels the classes from their position in place of their    |
         |             |           |           | append    | actual value namely, if classes=[7,9] it would return the   |
         |             |           |           | npl       | label 7->0, 9->1                                            |
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



    # TODO: train/test split customizable
    """

    default_params = {
        'binary':         0,
        'stochastic':     0,
        'classes':        (),  # no classes means all
        'vect':           False,
        'position_label': True,
        'subsampling':    None,
        'clip_high':      None,
        'clip_low':       None,
        'id_note':        None,
        'pm_one':         True,
    }

    classification = True  # true if

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = self._params['binary']
        self._pm_one = self._params['pm_one']

        # continuous mnist
        if self._binary_input == 0:
            default_data_dir = 'datasets/MNIST_data'
            self.data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir

            self._train_set_x, self._train_set_y, \
            self._validation_set_x, self._validation_set_y, \
            self._test_set_x, self._test_set_y = self.load_float_mnist(self.data_dir)
        # binary mnist
        else:
            default_data_dir = "datasets/MNIST_raw"
            self.data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir

            if self._params['stochastic'] == 1:
                self._train_set_x, self._train_set_y, \
                self._validation_set_x, self._validation_set_y, \
                self._test_set_x, self._test_set_y = self.load_binary_stochastic_mnist(self.data_dir)
            else:
                self._train_set_x, self._train_set_y, \
                self._validation_set_x, self._validation_set_y, \
                self._test_set_x, self._test_set_y = self.load_binary_det_mnist(self.data_dir, pm=self._pm_one)

        # filter classes
        if self._params['classes']:
            position_label = self._params['position_label']
            self._train_set_x, self._train_set_y = \
                self.class_filter(self._train_set_x, self._train_set_y, self._params['classes'], position_label)
            self._validation_set_x, self._validation_set_y = \
                self.class_filter(self._validation_set_x, self._validation_set_y, self._params['classes'],
                                  position_label)
            self._test_set_x, self._test_set_y = \
                self.class_filter(self._test_set_x, self._test_set_y, self._params['classes'], position_label)

        # choose a subset
        if self._params['subsampling']:
            self._train_set_x, self._train_set_y = \
                self.sub_sample(self._train_set_x, self._train_set_y, self._params['subsampling'])
            self._validation_set_x, self._validation_set_y = \
                self.sub_sample(self._validation_set_x, self._validation_set_y, self._params['subsampling'])
            self._test_set_x, self._test_set_y = \
                self.sub_sample(self._test_set_x, self._test_set_y, self._params['subsampling'])

        # clip
        clip_low = self._params['clip_low']
        clip_high = self._params['clip_high']
        if (clip_low is not None) or (clip_high is not None):
            m = clip_low if clip_low is not None else 0
            M = clip_high if clip_high is not None else 1
            self._train_set_x = np.clip(self._train_set_x, a_min=m, a_max=M)
            self._validation_set_x = np.clip(self._validation_set_x, a_min=m, a_max=M)
            self._test_set_x = np.clip(self._test_set_x, a_min=m, a_max=M)

        if self._params['vect']:
            self._train_set_x = self._train_set_x.reshape((-1, 784))
            self._validation_set_x = self._validation_set_x.reshape((-1, 784))
            self._test_set_x = self._test_set_x.reshape((-1, 784))

        else:
            self._train_set_x = self._train_set_x.reshape((-1, 28, 28, 1))
            self._validation_set_x = self._validation_set_x.reshape((-1, 28, 28, 1))
            self._test_set_x = self._test_set_x.reshape((-1, 28, 28, 1))

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                               'position_label', 'subsampling', 'clip_high', 'clip_low',
                               'data_dir', 'id_note', 'vect']  # all the admitted keys

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?

        MNIST.check_params_impl(params)

        id = 'MNIST'

        # binary or continuous
        id_binary = {
            0: '-c',
            1: '-d'}
        id += id_binary[params['binary']]

        # stochastic
        id += '-st' + str(params["stochastic"])

        # subclasses
        #
        if ('classes' in params) and (params['classes'] != ()):
            all_dg = list(range(10))  # list of available digits
            # check the list is a list of digits
            if params['classes'] is not None:
                if params['classes'] is not None:
                    assert (set(params['classes']) <= set(all_dg)), \
                        "classes contains labels not present in MNIST"
            id += ('-sc' + ''.join(map(str, params['classes'].sort())))  # append requested classes to the id

            # if position label is not activated
            if not params['position_label']:
                id += 'npl'

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

        if not params['pm_one']:
            id += '-pm%d' % int(params['pm_one'])

        return id

    @staticmethod
    def load_float_mnist(data_dir='datasets/MNIST_data', train_test_ratio=None):
        """ return MNIST data in a format suited for tensorflow.

            The script input_data is available under this URL:
            https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
        """
        assert (not train_test_ratio), "not implemented yet"

        from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        mnist = read_data_sets(data_dir, one_hot=False)

        train_set_x = mnist.train.images.astype(np.float32)
        validation_set_x = mnist.validation.images.astype(np.float32)
        test_set_x = mnist.test.images.astype(np.float32)

        # normalize data consistently (in case they would not already be)
        all_min, all_max = min_max_data_np([train_set_x, validation_set_x, test_set_x])
        train_set_x = normalize(train_set_x, all_min, all_max)
        validation_set_x = normalize(validation_set_x, all_min, all_max)
        test_set_x = normalize(test_set_x, all_min, all_max)

        train_set_y = mnist.train.labels.astype(np.int32)
        validation_set_y = mnist.validation.labels.astype(np.int32)
        test_set_y = mnist.test.labels.astype(np.int32)

        return train_set_x, train_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y

    @staticmethod
    def load_binary_stochastic_mnist(data_dir, train_test_ratio=None):
        """ stochastic

             the path is fixed
             see https://github.com/shuuki4/DRAW-tensorflow/blob/master/DRAW.py
             training process for MNIST

         """
        assert (not train_test_ratio), "not implemented yet"

        mnist_data_path = data_dir + "/mnist.pkl.gz"
        with gzip.open(mnist_data_path, "rb") as f:
            f.seek(0)
            train_set, validation_set, test_set = pickle.load(f, encoding="latin1")  # binarized MNIST
            train_set_x = train_set[0]
            train_set_y = train_set[1]
            validation_set_x = validation_set[0]
            validation_set_y = validation_set[1]
            test_set_x = test_set[0]
            test_set_y = test_set[1]
            # reshape
            train_set_x = np.reshape(train_set_x, [train_set_x.shape[0], 784])
            validation_set_x = np.reshape(validation_set_x, [validation_set_x.shape[0], 784])
            test_set_x = np.reshape(test_set_x, [test_set_x.shape[0], 784])

        train_set_x = train_set_x.astype(np.float32)
        validation_set_x = validation_set_x.astype(np.float32)
        test_set_x = test_set_x.astype(np.float32)

        train_set_y = train_set_y.astype(np.int32)
        validation_set_y = validation_set_y.astype(np.int32)
        test_set_y = test_set_y.astype(np.int32)

        return train_set_x, train_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y

    @staticmethod
    def load_binary_det_mnist(data_dir, train_test_ratio=None, pm=True):
        """ load the mnist from torchfiles in the path labels are not available
         """
        assert (not train_test_ratio), "not implemented yet"
        import torchfile

        mnist_data_path = data_dir
        if not os.path.exists(mnist_data_path):
            os.mkdir(mnist_data_path)
        # dataset from:
        # train: http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
        # validation: http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
        # test: http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat
        train_set_x = torchfile.load(mnist_data_path + "/train.t7")
        validation_set_x = torchfile.load(mnist_data_path + "/valid.t7")
        test_set_x = torchfile.load(mnist_data_path + "/test.t7")
        # no labels available here
        print('[WARNING] - This dataset has no lables')
        if pm:
            train_set_x = train_set_x.astype(np.float32) * 2 - 1
            validation_set_x = validation_set_x.astype(np.float32) * 2 - 1
            test_set_x = test_set_x.astype(np.float32) * 2 - 1
        else:
            train_set_x = train_set_x.astype(np.float32)
            validation_set_x = validation_set_x.astype(np.float32)
            test_set_x = test_set_x.astype(np.float32)
        return train_set_x, None, validation_set_x, None, test_set_x, None

    @staticmethod
    def sub_sample(data_set_x, data_set_y, subsampling):
        """
        return a value every "subsampling"

        :param data_set_x
        :param data_set_y
        :param subsampling: integer < dim(data_set)
        :return: dataset_x, dataset_y
        """

        len_data = len(data_set_x)
        reshuf_index_data = np.random.permutation(len_data)
        new_len_data = int(len_data / subsampling)

        data_set_x = data_set_x[reshuf_index_data[:new_len_data]]
        data_set_y = data_set_y[reshuf_index_data[:new_len_data]]

        return data_set_x, data_set_y

    def label_to_name(self, label):
        label_to_name_M_dict = {
            0: "Zero",
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Eight",
            9: "Nine"}
        return label_to_name_M_dict[label]

    @property
    def output_size(self):
        return 10 if self._params['classes'] == () else len(self._params['classes'])

    @property
    def color_images(self):
        return 0

    # generate iterator of random images in case of stochstic=1 and binary=1
    def dataset_map(self, dataset, datasets_tuple):

        # call parent
        dataset = super(MNIST, self).dataset_map(dataset, datasets_tuple)

        # sample from x if this is the case
        if self._binary_input and self._params['stochastic']:
            # the dataset is a tuple x, y
            if self._pm_one:
                return dataset.map(
                    lambda x, y: (tf.distributions.Bernoulli(probs=x, dtype=x.dtype).sample() * 2.0 - 1, y),
                    num_parallel_calls=NPROCS)
            else:
                return dataset.map(lambda x, y: (tf.distributions.Bernoulli(probs=x, dtype=x.dtype).sample(), y),
                                   num_parallel_calls=NPROCS)
        else:
            return dataset
