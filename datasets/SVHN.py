"""
Module for managing SVHN dataset
"""

import numpy as np

import os
import urllib
import tarfile

from .utils import normalize, min_max_data_np

from scipy.io import loadmat as load

import pdb

from .ImageDataset import ImageDataset

class SVHN(ImageDataset):
    """
    This class manage the dataset SVHN, properties of the datasets are uniquely determined
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

        default_data_dir = '/ssd_data/datasets/SVHN'
        self.data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir

        self._train_set_x, self._train_set_y, \
            self._validation_set_x,  self._validation_set_y, \
            self._test_set_x,  self._test_set_y = self.load_data(self.data_dir)

        #import pdb;pdb.set_trace()

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
            raise Exception("Not implemented")
            #size = no.prod(self._train_set_x.shape[1:])
            #self._train_set_x = self._train_set_x.reshape((-1,size))
            #self._validation_set_x = self._validation_set_x.reshape((-1,size))
            #self._test_set_x = self._test_set_x.reshape((-1,size))

        #else:
        #    self._train_set_x = self._train_set_x.reshape((-1, 28,28,1))
        #    self._validation_set_x = self._validation_set_x.reshape((-1, 28,28,1))
        #    self._test_set_x = self._test_set_x.reshape((-1, 28,28,1))

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                              'position_label', 'subsampling', 'clip_high', 'clip_low',
                              'data_dir', 'id_note','vect']  # all the admitted keys

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?

        SVHN.check_params_impl(params)

        id = 'SVHN'

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
    def load_data(save_to_path):

        try:
            os.stat(save_to_path)
        except:
            os.mkdir(save_to_path)

        #filename = save_to_path + "/train.tar.gz"
        #if not os.path.exists(save_to_path + "/train"):
        #    print("Downloading train.tar.gz ...")
        #    urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/train.tar.gz", filename)
        #    print("Extracting train.tar.gz ...")
        #    tar = tarfile.open(filename)
        #    tar.extractall(path=save_to_path)
        #    tar.close()

        # see http://ufldl.stanford.edu/housenumbers/

        filename = save_to_path + "/train_32x32.mat"
        if not os.path.exists(filename):
            print("Downloading train_32x32.mat ...")
            urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", filename)

        filename = save_to_path + "/test_32x32.mat"
        if not os.path.exists(filename):
            print("Downloading test_32x32.mat ...")
            urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", filename)

        # My uncomment
        # filename = save_to_path + "/extra_32x32.mat"
        # if not os.path.exists(filename):
        #     print("Downloading extra_32x32.mat ...")
        #     urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", filename)

        train = load(save_to_path + "/train_32x32.mat")
        test = load(save_to_path + "/test_32x32.mat")
        # My uncomment
        #extra = load(save_to_path + "/extra_32x32.mat")

        #train['y'].shape
        #(73257, 1)
        #train['X'].shape
        #(32, 32, 3, 73257)

        # change axis
        train['X'] = np.moveaxis(train['X'], 3, 0).astype(np.float32) / 255
        test['X'] = np.moveaxis(test['X'], 3, 0).astype(np.float32) / 255
        # My uncomment
        #extra['X'] = np.moveaxis(extra['X'], 3, 0).astype(np.float32) / 255

        train['X'] = 2 * train['X'] - 1
        test['X'] = 2 * test['X'] - 1
        # My uncomment
        #extra['X'] = 2 * extra['X'] - 1

        # points in the train_set
        n = 50000

        # extra is ignored, since the images are too simple
        return train['X'][:n], train['y'][:n][:,0]-1, train['X'][n:], train['y'][n:][:,0]-1, test['X'], test['y'][:,0]-1

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
