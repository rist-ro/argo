import os

os.environ['NUMEXPR_MAX_THREADS'] = '20'
from word_embedding.test.core.load_20newsgroup import read_20newsgroup

from .AlphaDatasetArr import AlphaDatasetArr

NPROCS = 4

class NewsgroupsArr(AlphaDatasetArr):
    default_params = {
        'split_seed' : 42,
    }

    def __init__(self, params):
        super().__init__(NewsgroupsArr.process_params(params))

        random_state = params.get('split_seed', 42)

        train_data, train_target, validation_data, validation_target, test_data, test_target = read_20newsgroup(
            ratio_datasets=[0.7, 0.15, 0.15], random_state=random_state)

        self._train_set_x, self._train_set_y = self._preprocess_arrays(train_data, train_target)
        self._validation_set_x, self._validation_set_y = self._preprocess_arrays(validation_data, validation_target)
        self._test_set_x, self._test_set_y = self._preprocess_arrays(test_data, test_target)

        self._set_shapes(n_samples_train = self._train_set_x.shape[0], n_labels = 20)


    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        NewsgroupsArr.check_params_impl(params)

        _id = '20newsgroupsArr'

        _id += AlphaDatasetArr.dataset_id(self, params)
        _id += '-s{:d}'.format(params['split_seed'])

        return _id






