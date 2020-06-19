import numpy as np
from scipy.io import loadmat

from datasets.ImageDataset import ImageDataset
from datasets.download import maybe_download_and_extract

TYPE16 = "16"
TYPE28 = "28"
TYPES = [TYPE16, TYPE28]


def rescale(x):
    return x * 2 - 1


class CaltechSilhouettes(ImageDataset):
    """THIS DATASET DOWNLOADS ITSELF
    This class manage the dataset CaltechSilhouettes, properties of the datasets are uniquely determined
    by the params dictionary

    https://people.cs.umass.edu/~marlin/data.shtml

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier


    """

    default_params = {
        #            'binary' : 0,
        'stochastic':  0,
        'vect':        False,
        'subsampling': None,
        'clip_high':   None,
        'clip_low':    None,
        'id_note':     None,
        'data_dir':    "/ssd_data/datasets/CaltechSilhouettes"
    }

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                               'position_label', 'subsampling', 'clip_high', 'clip_low',
                               'data_dir', 'id_note', 'vect']  # all the admitted keys

    classification = True  # true if

    def __init__(self, params):
        super().__init__(CaltechSilhouettes.process_params(params))

        self._id = self.dataset_id(self._params)
        self._binary_input = 0

        self._data_dir = self._params['data_dir']
        self._type = str(self._params['type'])
        if self._type == TYPE28:
            self._data_url = 'https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat'
        elif self._type == TYPE16:
            self._data_url = 'https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_16_split1.mat'
        else:
            raise Exception(
                "You chose '{}' which is an invalid type for Caltech Silhouettes. Only valid types: {}".format(
                    self._type, TYPES))
        self._data_extract = maybe_download_and_extract(self._data_url, self._data_dir)

        # Width and height of each image.
        self._picture_size = int(self._type)

        # Number of channels in each image, 3 channels: Red, Green, Blue.
        self._num_channels = 1

        self._n_params = len(self._params['params']) if 'params' in self._params else 0
        self._x_sample_shape = (self._picture_size, self._picture_size, self._num_channels)
        self._y_sample_shape = (self._n_params,)

        self._train_set_x, self._validation_set_x, self._test_set_x, self._train_set_y, self._validation_set_y, self._test_set_y = self.load_dataset_from_disk()

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

        dim = np.prod(self.x_shape)
        if self._params['vect']:
            self._train_set_x = self._train_set_x.reshape((-1, dim))
            self._validation_set_x = self._validation_set_x.reshape((-1, dim))
            self._test_set_x = self._test_set_x.reshape((-1, dim))

    def load_dataset_from_disk(self):
        """
            load the CaltechSilhouettes
            and its labels
        """

        dic = loadmat(self._data_extract)

        self._class_names = dic["classnames"]
        train_set_x = np.asarray(list(map(rescale, dic["train_data"].astype(np.float32)))).reshape(
            [-1, *self._x_sample_shape])
        train_set_y = dic["train_labels"].astype(np.float32)
        validation_set_x = np.asarray(list(map(rescale, dic["val_data"].astype(np.float32)))).reshape(
            [-1, *self._x_sample_shape])
        validation_set_y = dic["val_labels"].astype(np.float32)
        test_set_x = np.asarray(list(map(rescale, dic["test_data"].astype(np.float32)))).reshape(
            [-1, *self._x_sample_shape])
        test_set_y = dic["test_labels"].astype(np.float32)

        return train_set_x, validation_set_x, test_set_x, train_set_y, validation_set_y, test_set_y

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        CaltechSilhouettes.check_params_impl(params)

        id = 'CaltechSilhouettes'

        # binary or continuous
        # id_binary = {0:'-c',1:'-d'}
        # id += id_binary[params['binary']]

        # stochastic
        # id += '-st' + str(params["stochastic"])

        # types
        if ('type' in params) and params['type']:
            id += '-Ty' + str(params['type'])

        # subsampling
        if params['subsampling']:
            id += '-ss' + str(params['subsampling'])

        # clip
        # TODO The parameters of clip should be the values to which you clip
        clip_high = False
        if ('clip_high' in params) and params['clip_high']:
            id += '-cH'
            clip_high = True

        if ('clip_low' in params) and params['clip_low']:
            id += '-cL'
            if clip_high:
                id += "H"

        # id note (keep last)
        if ('id_note' in params) and params['id_note']:
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

        len_data = len(data_set_x)
        reshuf_index_data = np.random.permutation(len_data)
        new_len_data = int(len_data / subsampling)

        data_set_x = data_set_x[reshuf_index_data[:new_len_data]]
        data_set_y = data_set_y[reshuf_index_data[:new_len_data]]

        return data_set_x, data_set_y

    @property
    def output_size(self):
        return 0  # 10 if self._params['classes'] == () else len(self._params['classes'])

    @property
    def color_images(self):
        return 0

    @property
    def image_shape(self):
        return self._x_sample_shape  # the last number is the channel
