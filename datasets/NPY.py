import numpy as np

import os

from .ImageDataset import ImageDataset

import pdb

class NPY(ImageDataset):

    default_params = {
            'binary' : 0,
            #'stochastic' : 0,
            'classes' : (), # no classes means all
            'vect' : True,
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

        self._binary_input = self._params['binary']

        self._train_set_x = np.load(params["train_set_x"]) 
        self._test_set_x = np.load(params["test_set_x"]) 

        self._train_set_y = np.load(params["train_set_y"]) if "train_set_y" in params else None
        self._test_set_y = np.load(params["test_set_y"]) if "test_set_y" in params else None

        self._image_shape = self._train_set_x[0].shape
                
        self._train_set_x, self._train_set_y = self.preprocess_x_y(self._train_set_x, self._train_set_y)
        self._test_set_x, self._test_set_y = self.preprocess_x_y(self._test_set_x, self._test_set_y)

        
    implemented_params_keys = ['dataName', 'binary', 'classes',
                              'position_label', 'subsampling', 'clip_high', 'clip_low',
                              'data_dir', 'id_note','vect']  # all the admitted keys

    
    def preprocess_x_y(self, x, y):
        preprocessed_x = x.copy()
        preprocessed_y = y.copy() if (y is not None) else None
        
        # filter classes
        if self._params['classes']:
            position_label = self._params['position_label']
            preprocessed_x, preprocessed_y = \
                self.class_filter(preprocessed_x, preprocessed_y, self._params['classes'], position_label)

        # choose a subset
        if self._params['subsampling']:
            preprocessed_x, preprocessed_y = \
                self.sub_sample(preprocessed_x, preprocessed_y, self._params['subsampling'])
    
        #clip
        clip_low = self._params['clip_low']
        clip_high = self._params['clip_high']
        if (clip_low is not None) or (clip_high is not None):
            m = clip_low if clip_low is not None else 0
            M = clip_high if clip_high is not None else 1
            preprocessed_x = np.clip(preprocessed_x, a_min=m, a_max=M)
                
        if self._params['vect']:
            length = np.prod(self._image_shape)
            preprocessed_x = preprocessed_x.reshape((-1,length))

        return preprocessed_x, preprocessed_y
    
    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?

        NPY.check_params_impl(params)

        id = 'NPY-' + params["id"]

        # binary or continuous
        id_binary = {0:'-c',1:'-d'}
        id += id_binary[params['binary']]

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
                        "classes contains labels not present in NPY"
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

    @property
    def input_size(self):
        return np.prod(self._train_set_x[0].shape)

    @property
    def output_size(self):
        # TODO check if this is ok or not
        return self.y_shape if self._params['classes'] == () else len(self._params['classes'])

    @property 
    def color_images(self):
        return 0

    # TO BE REMOVED
    @property
    def image_shape(self):
        return self._image_shape

