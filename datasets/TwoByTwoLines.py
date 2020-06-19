"""
Module for managing the IRIS dataset
"""

import numpy as np

from .Dataset import Dataset

DATASET_SIZE = 15


class TwoByTwoLines(Dataset):
    """
    This class manage the dataset
    , properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """

    default_params = {
        'size':   2,
        'pm_one': True
    }

    classification = False  # true if

    implemented_params_keys = ['dataName']  # all the admitted keys

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = True
        self._size = params['size']

        self._pm_one = params['pm_one']

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_data(self._size)

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        TwoByTwoLines.check_params_impl(params)

        id = '2by2'
        if params['size'] != 2:
            id += '-sz%d' % params['size']
        if not params['pm_one']:
            id += '-pm%d' % int(params['pm_one'])
        return id

    def load_data(self, n):
        ver, hor = self.get_horizontal_pattern(n=n)

        if self._pm_one:
            ver, hor = ver * 2 - 1, hor * 2 - 1

        dataset = []

        dataset += list(hor) * DATASET_SIZE
        dataset += list(ver) * DATASET_SIZE
        #
        # dataset = np.ones((4*DATASET_SIZE,4))
        dataset = np.asarray(dataset)

        len_of_dataset = len(dataset)
        print("LEN OF DATASET!!! ", len_of_dataset)
        dataset = dataset.reshape((len_of_dataset, n, n, 1))

        perm = np.random.permutation(len_of_dataset)
        dataset = dataset[perm].astype(np.float32)
        partition = int(len_of_dataset / 6)
        # extra is ignored, since the images are too simple
        return dataset[:len_of_dataset - partition], \
               None, \
               dataset[len_of_dataset - partition:], \
               None, \
               dataset[len_of_dataset - partition:], \
               None

    '''
    @property
    def output_size(self):
        return 10 if self._params['classes'] == () else len(self._params['classes'])
    '''

    @property
    def color_images(self):
        return 0

    @property
    def image_shape(self):
        return (self._size, self._size, 1)  # the last number is the channel

    # Likelihood always returns the patterns as 0,1, not -1,1
    @property
    def likelihood(self):
        ver, hor = TwoByTwoLines.get_horizontal_pattern(n=self._size)
        patterns = [*ver, *hor]
        len_p = len(patterns)
        patterns_and_likelihoods = [(i.ravel(), 1 / len_p) for i in patterns]

        return patterns_and_likelihoods

    @staticmethod
    def get_horizontal_pattern(n=3):
        def get_numbers(n):
            return np.arange(1, 2 ** n - 1)

        x = get_numbers(n)
        x = [('{0:0' + str(n) + 'b}').format(i) for i in x]
        x = [np.tile(i, n) for i in x]

        vertical_lines = np.asarray([list(map(lambda j: [int(k) for k in j], i)) for i in x])
        horizontal_lines = np.transpose(vertical_lines, [0, 2, 1])
        return vertical_lines, horizontal_lines
