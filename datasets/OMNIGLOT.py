"""
Module for managing OMNIGLOT dataset
"""

import numpy as np

import os.path
import urllib.request

#from .Dataset import Dataset    ## TODO  Check wiythout point
from .ImageDataset import ImageDataset
from scipy.io import loadmat

import pdb

class OMNIGLOT(ImageDataset):
    """
    This class manage the dataset OMNIGLOT, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default one. It then return a unique id identifier

    Parameters
    ---------

    params : dict
        dictionary that can contain
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | params Key  | values    | default v |  id short | Short description                                           |
         +=============+===========+===========+===========+=============================================================+
         |   binary    |  0,1      |  0        | "-c", "-d"| load continuous or binary OMNIGLOT                          |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         |  stochastic |  0,1      |  0        |   "-st"   | sample using Bernoulli from the continuous OMNIGLOT after every|
         |             |           |           |           | epoch during training, see IWAE and LVAE papers that claim  |
         |             |           |           |           | this techniques reduces overfitting; this function only     |
         |             |           |           |           | loads continuous OMNIGLOT to be used later                     |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | data_dir    | path str  | None      |           | path of the dataset. In some cases cannot be set            |
         |             |           |           |           | (non binary mnist only)                                     |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | classes     | e.g.      |           |           | select a list of subclasses of digits, e.g. [0,1]           |
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
            'binary' : 0,
            'stochastic' : 0,
            'classes' : (), # no classes means all
            'position_label' : True,
            'subsampling' : None,
            'clip_high' :None,
            'clip_low' : None,
            'id_note' : None
        }

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = self._params['binary']

        self.data_dir = "datasets/OMNIGLOT_data"

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self._train_set_x, self._validation_set_x, self._train_set_y, self._validation_set_y = self.load_data()

        # continuous omniglot
        if self._binary_input==0 or (self._binary_input==0 and self._params['stochastic']==1):
            dtype = 'f'
        else:
            dtype = 'i'


        # filter classes
        if self._params['classes']:
            position_label = self._params['position_label']
            self._train_set_x, self._train_set_y  = \
                self.class_filter(self._train_set_x, self._train_set_y, self._params['classes'], position_label)
            self._validation_set_x, self._validation_set_y = \
                self.class_filter(self._validation_set_x, self._validation_set_y, self._params['classes'], position_label)

        # choose a subset
        if self._params['subsampling']:
            self._train_set_x, self._train_set_y = \
                self.sub_sample(self._train_set_x, self._train_set_y, self._params['subsampling'])
            self._validation_set_x, self._validation_set_y = \
                self.sub_sample(self._validation_set_x, self._validation_set_y, self._params['subsampling'])

        #clip
        clip_low = self._params['clip_low']
        clip_high = self._params['clip_high']
        if (clip_low is not None) or (clip_high is not None):
            m = clip_low if clip_low is not None else 0
            M = clip_high if clip_high is not None else 1
            self._train_set_x = np.clip(self._train_set_x, a_min=m, a_max=M)
            self._validation_set_x = np.clip(self._validation_set_x, a_min=m, a_max=M)


    def load_data(self):
        fileName = self.data_dir + '/chardata.mat'
        if not os.path.isfile(fileName):
            # see https://github.com/casperkaae/parmesan/blob/master/parmesan/datasets.py
            origin = (
                'https://github.com/yburda/iwae/raw/'
                'master/datasets/OMNIGLOT/chardata.mat'
            )
            print('Downloading data from %s' % origin)

            data = urllib.request.urlretrieve(origin, fileName)

        mat = loadmat(fileName)

        _train_set_x = self.resize_to_pic(np.array(mat['data'].T, dtype='f'))
        _test_set_x = self.resize_to_pic(np.array(mat['testdata'].T, dtype='f'))
        _train_set_y = np.argmax(np.array(mat['target'].T, dtype='f'), axis=1)
        _test_set_y = np.argmax(np.array(mat['testtarget'].T, dtype='f'), axis=1)

        return (_train_set_x, _test_set_x, _train_set_y, _test_set_y)



        implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                                   'position_label', 'subsampling', 'clip_high', 'clip_low',
                                   'data_dir', 'id_note']  # all the admitted keys

    def resize_to_pic(self, ds):
        orig_shape = ds.shape
        return ds.reshape(orig_shape[0], *self.image_shape)


    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?
        OMNIGLOT.check_params_impl(params)

        id = 'OMNIGLOT'

        # binary or continuous
        id_binary = {0:'-c',1:'-d'}
        id += id_binary[params['binary']]

        # stochastic
        id += '-st' + str(params["stochastic"])

        # subclasses
        # TODO: argo may split the list of classes into subprocesses, easy solution: write a string in place of a list.
        #
        if ('classes' in params.keys()) and (params['classes'] != ()):
            all_dg = list(range(50)) # list of available digits
            # check the list is a list of digits
            if params['classes'] is not None:
                if params['classes'] is not None:
                    assert (set(params['classes']) <= set(all_dg)), \
                        "classes contains labels not present in OMNIGLOT"
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

    '''
    def get_data_dict(self):

        if not self._binary_input or (self.params['binary'] and not self.params['stochastic']):
            ds["train_set_y"] = self._train_set_y
            ds["test_set_y"] = self._test_set_y

        return ds
    '''

    @property
    def input_size(self):
        return 784

    @property
    def output_size(self):
        return 50 if self._params['classes'] == [] else len(self._params['classes'])

    @property 
    def color_images(self):
        return 0

    @property
    def image_shape(self):
        return (28,28,1) # 1 is the number of channels
