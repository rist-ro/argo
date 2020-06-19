"""
Module for managing the IRIS dataset
"""

import numpy as np

from sklearn import datasets

import pdb

from .Dataset import Dataset

class IRIS(Dataset):
    """
    This class manage the dataset IRIS, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """

    default_params = {
            'stochastic' : 0,
            'classes' : (), # no classes means all
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

        self._train_set_x, self._train_set_y, \
            self._validation_set_x,  self._validation_set_y, \
            self._test_set_x,  self._test_set_y = self.load_data()
        
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

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                              'position_label', 'subsampling',
                               'id_note']  # all the admitted keys

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?

        IRIS.check_params_impl(params)

        id = 'IRIS'

        # stochastic
        id += '-st' + str(params["stochastic"])

        # subclasses
        if ('classes' in params) and (params['classes'] != ()):
            all_dg = list(range(10)) # list of available digits
            # check the list is a list of digits
            if params['classes'] is not None:
                if params['classes'] is not None:
                    assert (set(params['classes']) <= set(all_dg)), \
                        "classes contains labels not present in SVHN"
            id += ('-sc' +  ''.join(map(str, params['classes'].sort())))  # append requested classes to the id

            # if position label is not activated
            if not params['position_label']:
                id+='npl'

        # subsampling
        if params['subsampling']:
            id += '-ss'+str(params['subsampling'])

        # id note (keep last)
        if params['id_note']:
            id += params['id_note']

        return id

    @staticmethod
    def load_data():

        # see https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
        iris = datasets.load_iris()

        perm = np.random.permutation(len(iris.target))
        
        data = iris.data[perm].astype(np.float32)
        target = iris.target[perm]

        # points in the train_set
        n = 120
        
        # extra is ignored, since the images are too simple
        return data[:n], target[:n], data[n:], target[n:], data[n:], target[n:]

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
