"""
Module for managing FashionMNIST dataset
"""

import numpy as np
from .utils import normalize, min_max_data_np

import os

from tensorflow.examples.tutorials.mnist import input_data

import pdb

from .ImageDataset import ImageDataset

class FashionMNIST(ImageDataset):
    """
    This class manage the dataset FashionMNIST, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """

    default_params = {
            'stochastic' : 0,
            'classes' : (), # no classes means all
            'vect' : False,
            'position_label' : True,
            'subsampling' : None,
            'clip_high' :None,
            'clip_low' : None,
            'id_note' : None
        }


    classification = True # true if

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = False

        default_data_dir = '/data1/datasets/FashionMNIST'
        self.data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir
        
        self._train_set_x, self._train_set_y, \
            self._validation_set_x,  self._validation_set_y, \
            self._test_set_x,  self._test_set_y = self.load_data()

        #normalize data
        all_min, all_max = min_max_data_np([self._train_set_x, self._validation_set_x, self._test_set_x])
        self._train_set_x = normalize(self._train_set_x, all_min, all_max)
        self._validation_set_x = normalize(self._validation_set_x, all_min, all_max)
        self._test_set_x = normalize(self._test_set_x, all_min, all_max)


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
            self._train_set_x = self._train_set_x.reshape((-1,784))
            self._validation_set_x = self._validation_set_x.reshape((-1,784))
            self._test_set_x = self._test_set_x.reshape((-1,784))

        else:
            self._train_set_x = self._train_set_x.reshape((-1, 28,28,1))
            self._validation_set_x = self._validation_set_x.reshape((-1, 28,28,1))
            self._test_set_x = self._test_set_x.reshape((-1, 28,28,1))

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                              'position_label', 'subsampling', 'clip_high', 'clip_low',
                              'data_dir', 'id_note','vect']  # all the admitted keys

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?

        FashionMNIST.check_params_impl(params)

        id = 'FashionMNIST'

        # stochastic
        id += '-st' + str(params["stochastic"])

        # subclasses
        if ('classes' in params) and (params['classes'] != ()):
            all_dg = list(range(10)) # list of available digits
            # check the list is a list of digits
            if params['classes'] is not None:
                if params['classes'] is not None:
                    assert (set(params['classes']) <= set(all_dg)), \
                        "classes contains labels not present in FashionMNIST"
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

        return id

    @staticmethod
    def load_data():
        # this needs to be rewritten based on the warning you get in TF
        data = input_data.read_data_sets('/ssd_data/datasets/fashionMNIST',
                                         source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

        return data.train.images, data.train.labels,\
               data.validation.images, data.validation.labels,\
               data.test.images, data.test.labels

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
        label_to_name_F_dict = {0: "T-shirt/top",
                                1: "Trouser",
                                2: "Pullover",
                                3: "Dress",
                                4: "Coat",
                                5: "Sandal",
                                6: "Shirt",
                                7: "Sneaker",
                                8: "Bag",
                                9: "Ankle boot"}
        return label_to_name_F_dict[label]

    '''
    @property
    def output_size(self):
        return 10 if self._params['classes'] == () else len(self._params['classes'])

    @property
    def color_images(self):
        return 0

    @property
    def image_shape(self):
        return (28,28,1) # the last number is the channel
    '''
