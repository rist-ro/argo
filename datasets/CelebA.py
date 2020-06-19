"""
Module for managing CelebA dataset
"""
import json

import numpy as np

import pandas as pd

import os

#import gzip
#import pickle

from .ImageDataset import ImageDataset

ALL_LABELS = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
               'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
               'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
               'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
               'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
               'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
               'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
              'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
              'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
              'Wearing_Necktie', 'Young', 'image_id.1', 'x_1', 'y_1', 'width',
              'height']

class CelebA(ImageDataset):
    """
    This class manage the dataset CelebA, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier


    # TODO: train/test split customizable
    """

    default_params = {
        #            'binary' : 0,
        'stochastic' : 0,
        'vect' : False,
        'subsampling' : None,
        'clip_high' :None,
        'clip_low' : None,
        'id_note' : None,
        'test_and_validation' : 30000,
        'data_dir': "/ssd_data/datasets/CelebA"
    }

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                               'position_label', 'subsampling', 'clip_high', 'clip_low',
                               'data_dir', 'id_note', 'vect']  # all the admitted keys

    classification = False # true if

    def __init__(self, params):
        super().__init__(CelebA.process_params(params))

        self._id = self.dataset_id(self._params)
        self._binary_input = 0

        self._data_dir = self._params['data_dir']

        # get info from the metadata
        self.labels_df = self.read_labels()

        # Width and height of each image.
        self._picture_size = 64

        # Number of channels in each image, 3 channels: Red, Green, Blue.
        self._num_channels = 3

        self._n_params = len(self._params['params']) if 'params' in self._params else 0
        self._x_sample_shape = (self._picture_size, self._picture_size, self._num_channels)
        self._y_sample_shape = (self._n_params,)

        self._var_params = self.get_var_params(self._params['params']) if 'params' in self._params else []

        self._train_set_x, self._validation_set_x, self._test_set_x, self._train_set_y, self._validation_set_y, self._test_set_y = self.load_dataset_from_disk()

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

        dim = np.prod(self.x_shape)
        if self._params['vect']:
            raise Exception("I refuse to flatten CelebA! Take me as I am. I am a dataset with my own dignity and my own channels. All the Best.") # Riccardo
            self._train_set_x = self._train_set_x.reshape((-1,dim))
            self._validation_set_x = self._validation_set_x.reshape((-1,dim))
            self._test_set_x = self._test_set_x.reshape((-1,dim))

    def read_labels(self):
        """

        :return:
        """
        #load csv file
        csv_filename = self._data_dir + '/concatenated_labels.csv'
        label_df = pd.read_csv(csv_filename)

        return label_df

    def load_dataset_from_disk(self):
        """
            load the CelebA
            and its labels
        """
        X = np.load(self._data_dir + "/all.npy").astype(np.float32) / 255
        n_files = X.shape[0]

        Y = np.asarray(self.labels_df[self._var_params])

        randomized_dataset_index = np.random.permutation(n_files)
        X = X[randomized_dataset_index]
        Y = Y[randomized_dataset_index]

        test_split = n_files - self._params['test_and_validation']
        validation_split = test_split - self._params['test_and_validation']

        trainVal, test_set_x = X[:test_split,:], X[test_split:,:]
        train_set_x, validation_set_x = trainVal[:validation_split,:], trainVal[validation_split:,:]

        trainVal, test_set_y = Y[:test_split, :], Y[test_split:, :]
        train_set_y, validation_set_y = trainVal[:validation_split, :], trainVal[validation_split:, :]

        return train_set_x, validation_set_x, test_set_x, train_set_y, validation_set_y, test_set_y

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?

        CelebA.check_params_impl(params)

        id = 'CelebA'

        # binary or continuous
        #id_binary = {0:'-c',1:'-d'}
        #id += id_binary[params['binary']]

        # stochastic
        #id += '-st' + str(params["stochastic"])

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
        if ('clip_high' in params) and  params['clip_high'] :
            id += '-cH'
            clip_high = True

        if ('clip_low' in params) and params['clip_low'] :
            id += '-cL'
            if clip_high:
                id += "H"

        # id note (keep last)
        if ('id_note' in params) and params['id_note']:
            id += params['id_note']

        return id

    def get_var_params(self, param):
        var_params = np.intersect1d(ALL_LABELS, param)

        assert len(var_params) == len(param), "It seems like you might have a mistake in you label name"
        return var_params



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



    '''
    def get_data_dict(self):

        if not self._binary_input or (self.params['binary'] and not self.params['stochastic']):
            ds["train_set_y"] = self._train_set_y
            ds["test_set_y"] = self._test_set_y

        return ds
    '''

    '''
    # this should be removed
    @property
    def input_size(self):
        return 64*64
    '''

    @property
    def output_size(self):
        return 0 # 10 if self._params['classes'] == () else len(self._params['classes'])

    @property
    def color_images(self):
        return 1

    '''
    # this should be automatic
    @property
    def image_shape(self):
        return self._input_shape # the last number is the channel
    '''
