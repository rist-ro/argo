import os
import tensorflow_datasets as tfds

os.environ['NUMEXPR_MAX_THREADS'] = '20'

from .AlphaDatasetArr import AlphaDatasetArr

NPROCS = 4

class IMDBReviewsArr(AlphaDatasetArr):
    default_params = {
    }

    def __init__(self, params):
        super().__init__(IMDBReviewsArr.process_params(params))

        train_data, train_target, validation_data, validation_target, test_data, test_target = self._read_imdb()

        self._train_set_x, self._train_set_y = self._preprocess_arrays(train_data, train_target)
        self._validation_set_x, self._validation_set_y = self._preprocess_arrays(validation_data, validation_target)
        self._test_set_x, self._test_set_y = self._preprocess_arrays(test_data, test_target)

        self._set_shapes(n_samples_train = self._train_set_x.shape[0], n_labels = 2)


    def dataset_id(self, params):
        """
        This method interprets the parameters and generate an id
        """

        IMDBReviewsArr.check_params_impl(params)

        _id = 'IMDBReviewsArr'

        _id += AlphaDatasetArr.dataset_id(self, params)

        return _id

    def _read_imdb(self):
        train_dataset, info = tfds.load('imdb_reviews/plain_text', split="train", as_supervised=True, with_info=True)
        # n_samples_train = info.splits['train'].num_examples
        # val/test equal split
        # n_samples_validation = info.splits['test'].num_examples / 2.
        # n_samples_test = info.splits['test'].num_examples / 2.

        # # new S3 API, still not supported by imdb reviews
        # validation_dataset = tfds.load('imdb_reviews/plain_text', split="test[:50%]", as_supervised=True)
        # test_dataset = tfds.load('imdb_reviews/plain_text', split="test[-50%:]", as_supervised=True)

        # legacy API
        validation_split, test_split = tfds.Split.TEST.subsplit(k=2)
        validation_dataset = tfds.load('imdb_reviews/plain_text', split=validation_split, as_supervised=True)
        test_dataset = tfds.load('imdb_reviews/plain_text', split=test_split, as_supervised=True)

        train_data, train_target = zip(*[excerpt for excerpt in tfds.as_numpy(train_dataset)])
        validation_data, validation_target = zip(*[excerpt for excerpt in tfds.as_numpy(validation_dataset)])
        test_data, test_target = zip(*[excerpt for excerpt in tfds.as_numpy(test_dataset)])
        return train_data, train_target, validation_data, validation_target, test_data, test_target





