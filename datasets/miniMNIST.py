import numpy as np

import tensorflow as tf

import os

import gzip
import pickle

from .Dataset import NPROCS

import pdb

from .MNIST import MNIST

from .utils import normalize, min_max_data_np


class miniMNIST(MNIST):

    default_params = {
            'binary' : 0,
            'stochastic' : 0,
            'classes' : (), # no classes means all
            'vect' : False,
            'position_label' : True,
            'subsampling' : None,
            'clip_high' :None,
            'clip_low' : None,
            'id_note' : None,
            'pm_one':  True,
        }


    classification = True # true if

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
                self._validation_set_x,  self._validation_set_y, \
                self._test_set_x,  self._test_set_y = self.load_float_mnist(self.data_dir)
        # binary mnist
        else:
            raise Exception("not implemented")

            default_data_dir = "datasets/miniMNIST_raw"
            self.data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir

            if self._params['stochastic'] == 1:
                self._train_set_x, self._train_set_y, \
                    self._validation_set_x, self._validation_set_y, \
                    self._test_set_x, self._test_set_y = self.load_binary_stochastic_mnist(self.data_dir)
            else:
                self._train_set_x, self._train_set_y, \
                    self._validation_set_x, self._validation_set_y, \
                    self._test_set_x, self._test_set_y = self.load_binary_det_mnist(self.data_dir)

        # filter classes
        if self._params['classes']:
            position_label = self._params['position_label']
            self._train_set_x, self._train_set_y  = \
                self.class_filter(self._train_set_x, self._train_set_y, self._params['classes'], position_label)
            self._validation_set_x, self._validation_set_y = \
                self.class_filter(self._validation_set_x, self._validation_set_y, self._params['classes'], position_label)
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

        #clip
        clip_low = self._params['clip_low']
        clip_high = self._params['clip_high']
        if (clip_low is not None) or (clip_high is not None):
            m = clip_low if clip_low is not None else 0
            M = clip_high if clip_high is not None else 1
            self._train_set_x = np.clip(self._train_set_x, a_min=m, a_max=M)
            self._validation_set_x = np.clip(self._validation_set_x, a_min=m, a_max=M)
            self._test_set_x = np.clip(self._test_set_x, a_min=m, a_max=M)

        if self._params['vect']:
            self._train_set_x = self._train_set_x.reshape((-1,196))
            self._validation_set_x = self._validation_set_x.reshape((-1,196))
            self._test_set_x = self._test_set_x.reshape((-1,196))

        else:
            self._train_set_x = self._train_set_x.reshape((-1, 14,14,1))
            self._validation_set_x = self._validation_set_x.reshape((-1, 14,14,1))
            self._test_set_x = self._test_set_x.reshape((-1, 14,14,1))

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                              'position_label', 'subsampling', 'clip_high', 'clip_low',
                              'data_dir', 'id_note','vect']  # all the admitted keys

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?

        miniMNIST.check_params_impl(params)

        id = 'miniMNIST'

        # binary or continuous
        id_binary = {0:'-c',1:'-d'}
        id += id_binary[params['binary']]

        # stochastic
        id += '-st' + str(params["stochastic"])

        # subclasses
        #
        if ('classes' in params) and (params['classes'] != ()):
            all_dg = list(range(10)) # list of available digits
            # check the list is a list of digits
            if params['classes'] is not None:
                if params['classes'] is not None:
                    assert (set(params['classes']) <= set(all_dg)), \
                        "classes contains labels not present in MNIST"
            id += ('-sc' +  ''.join(map(str, params['classes'].sort())))  # append requested classes to the id

            # if position label is not activated
            if not params['position_label']:
                id+='npl'

        # subsampling
        if params['subsampling']:
            id += '-ss'+str(params['subsampling'])

        # clip
        # TODO The parameters of clip should be the values to which you clip
        clip_high = False
        if  params['clip_high'] :
            id += '-cH'
            clip_high = True

        if params['clip_low'] :
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
        assert(not train_test_ratio), "not implemented yet"


        from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        mnist = read_data_sets(data_dir, one_hot=False)

        '''
        # old
        train_set_x = mnist.train.images.astype(np.float32)
        validation_set_x = mnist.validation.images.astype(np.float32)
        test_set_x = mnist.test.images.astype(np.float32)
        '''

        # new
        mini_data_dir = '/ssd_data/datasets/miniMNIST'
        train_set_x = np.load(mini_data_dir + '/train_set_x.npy').astype(np.float32)
        validation_set_x = np.load(mini_data_dir + '/validation_set_x.npy').astype(np.float32)
        test_set_x = np.load(mini_data_dir + '/test_set_x.npy').astype(np.float32)

        # normalize data consistently (in case they would not already be)
        all_min, all_max = min_max_data_np([train_set_x, validation_set_x, test_set_x])
        train_set_x = normalize(train_set_x, all_min, all_max)
        validation_set_x = normalize(validation_set_x, all_min, all_max)
        test_set_x = normalize(test_set_x, all_min, all_max)

        train_set_y = mnist.train.labels.astype(np.int32)
        validation_set_y = mnist.validation.labels.astype(np.int32)
        test_set_y = mnist.test.labels.astype(np.int32)

        return train_set_x, train_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y

