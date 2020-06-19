"""
Module for managing the IRIS dataset
"""

import numpy as np

import pdb

from .Dataset import Dataset

DATASET_SIZE = 1000

freequent_patterns = [
    [-1, -1, 1, -1, -1, 1, -1, -1, 1],
    [-1, 1, -1, -1, 1, -1, -1, 1, -1],
    [1, -1, -1, 1, -1, -1, 1, -1, -1],
    [1, -1, 1, 1, -1, 1, 1, -1, 1],
    [1, 1, -1, 1, 1, -1, 1, 1, -1],
    [-1, 1, 1, -1, 1, 1, -1, 1, 1],

]
less_freequent_patterns = [
    [-1, -1, -1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, 1, 1, 1],
    [-1, -1, -1, 1, 1, 1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, -1, -1, -1],
    [1, 1, 1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, -1, -1, -1, 1, 1, 1],
]
freequent_patterns_freqquency = 2. / 18.
less_freequent_patterns_freqquency = 1. / 18.


class ThreeByThreeLines(Dataset):
    """
    This class manage the dataset ThreeByThreeLines, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """

    default_params = {
        'pm_one': True
    }

    classification = False  # true if

    implemented_params_keys = ['dataName']  # all the admitted keys

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = True

        self._pm_one = params['pm_one']

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_data()

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        ThreeByThreeLines.check_params_impl(params)

        id = '3by3'
        if not params['pm_one']:
            id += '-pm%d' % int(params['pm_one'])

        return id

    @staticmethod
    def generate_dataset(nr_of_samples):
        # The dataset is defined in the Tutorial on Helmholtz Machines by Kevin G Kirby
        dataset = []

        nr_of_freequent = 2 * nr_of_samples
        nr_of_infreequent = nr_of_samples

        dataset += freequent_patterns * nr_of_freequent
        dataset += less_freequent_patterns * nr_of_infreequent

        dataset = np.asarray(dataset)

        len_of_dataset = len(dataset)

        # make them 3*3 pictures
        return dataset.reshape((len_of_dataset, 3, 3, 1))

    def load_data(self):
        # see https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
        threeByThree = self.generate_dataset(DATASET_SIZE)

        len_of_dataset = len(threeByThree)

        perm = np.random.permutation(len_of_dataset)

        data = threeByThree[perm].astype(np.float32)

        data = self._pm_cast(data)

        partition = int(len_of_dataset / 6)
        # extra is ignored, since the images are too simple
        return data[:len_of_dataset - partition], None, data[len_of_dataset - partition:], None, data[
                                                                                                 len_of_dataset - partition:], None

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
        return (3, 3, 1)  # the last number is the channel

    # Likelihood always returns the patterns as 0,1, not -1,1
    @property
    def likelihood(self):
        patterns_and_likelihoods = [((np.asarray(i)+1)/2, freequent_patterns_freqquency) for i in freequent_patterns]
        patterns_and_likelihoods += [((np.asarray(i)+1)/2, less_freequent_patterns_freqquency) for i in less_freequent_patterns]

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
        return vertical_lines * 2 - 1, horizontal_lines * 2 - 1

    def _pm_cast(self, lis):
        if self._pm_one:
            return lis
        else:
            return (np.asarray(lis) + 1) / 2
