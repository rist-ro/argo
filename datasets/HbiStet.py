"""
Module for managing WavFromGeneralMidi dataset
"""

from functools import partial

import numpy as np
import tensorflow as tf

from datasets.AudioDataset import AudioDataset
from .Dataset import TRAIN, VALIDATION, TEST

LABEL = "label"
NAME = "name"
AUDIO = "audio"

ALL_LABELS_IMPLEMENTED = [LABEL,
                          NAME,
                          AUDIO]

ALL_LABELS = ALL_LABELS_IMPLEMENTED

LABEL_TO_INT_DICT = {
    "No": '4',
    "extrahls": '3',
    "artifact": '2',
    "murmur": '1',
    "normal": '0',
}

class HbiStet(AudioDataset):
    """
    This class manage the dataset Heartbeat iStetoscope, properties of the datasets are uniquely determined
    by the params dictionary

    https://www.kaggle.com/kinguistics/heartbeat-sounds

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """

    default_params = {
        'test_and_validation': 8515,
        # 'data_dir':            "temp_data2",
        'data_dir':            "/ssd_data/datasets/HB",
        'shuffle_buffer':      60000,
        "anomaly_detection": False
    }

    classification = False  # true if

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(self._params)

        self._data_dir = self._params['data_dir']

        # Width and height of each image.
        self._sample_lenght = 396900 + 412  # 128 padded 0s  # used to be 88384 for sample rate 44100
        self._sample_rate = 44100
        self._label = (self._params['label'] if self._params['label'] in ALL_LABELS_IMPLEMENTED else None)

        # self._n_labels = len(self._params['features']) if 'features' in self._params else 0
        self._x_sample_shape_train = [self._crop_length_train, 1]
        self._x_sample_shape_eval = [self._sample_lenght, 1]
        self._y_sample_shape = None  # (self._n_labels,)

        self._n_labels = len(LABEL_TO_INT_DICT)

        # self._var_labels = self.get_var_labels(self._params['features']) if 'features' in self._params else []

        # self._train_set_x, self._validation_set_x, self._test_set_x, self._train_set_y, self._validation_set_y, self._test_set_y = [None] * 6

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        HbiStet.check_params_impl(params)

        _id = 'HbiStet'

        # TODO I know this is bad but if id is a static method and not an object method I have no idea how to handle this better,
        # TODO soon we should refactor the id to be multilayer of abstraction as in Networks and Models.
        if params['shuffle_buffer'] != HbiStet.default_params['shuffle_buffer']:
            _id += '-sh%.2e' % params['shuffle_buffer']

        if params.get('anomaly_detection', False):
            _id += '-anomaly_detection'

        return _id

    def get_var_labels(self, param):
        var_params = np.intersect1d(ALL_LABELS, param)

        assert len(var_params) == len(param), "It seems like you might have a mistake in you label name"
        return var_params

    @staticmethod
    def _parse_function(example_proto, y_label=None):
        features = {
            LABEL: tf.FixedLenFeature([], dtype=tf.string),
            NAME:  tf.FixedLenFeature([], dtype=tf.string),
            AUDIO: tf.VarLenFeature(dtype=tf.float32), #max 396900
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # Expand dims for channels
        audio = tf.expand_dims(tf.concat(parsed_features[AUDIO].values, axis=0), axis=-1)
        label = AudioDataset.str_label_to_int(parsed_features[y_label], LABEL_TO_INT_DICT)

        if y_label:
            return (audio, label,)
        else:
            return (audio,)

    def get_dataset_iterator(self, batch_size, dataset_str, shuffle, repeat, augment, perturb):

        is_perturbed = False
        filename = ""

        # create Dataset objects using the data previously downloaded
        if dataset_str == TRAIN:
            filename = self._data_dir + "/train3-istet-float-shuffled.tfrecords"

        elif dataset_str == VALIDATION:
            filename = self._data_dir + "/validation3-istet-float-shuffled.tfrecords"

        elif dataset_str == TEST:
            filename = self._data_dir + "/test_all-2.tfrecords"

        else:
            raise Exception("dataset not recognized (accepted values are: train, validation and test)")

        # CREATE TF DATASET with map and py_func
        dataset = tf.data.TFRecordDataset([filename])

        NPROCS = 20
        if self._label:
            parse_func = lambda x: self._parse_function(x, y_label=self._label)
        else:
            parse_func = self._parse_function

        dataset = dataset.map(parse_func,
                              num_parallel_calls=NPROCS)

        # caching before shuffling and batching for super cow speed
        dataset = dataset.cache()

        # PREPROCESS DATA (AUGMENT IF NEEDED)
        if augment:
            dataset = dataset.map(partial(self._crop_element, is_perturbed, self._crop_length_train), num_parallel_calls=NPROCS) \
                .map(self.augment_element, num_parallel_calls=NPROCS)
        if dataset_str == TEST:
            dataset = dataset.map(partial(self._crop_element, is_perturbed, self._crop_length_train), num_parallel_calls=NPROCS) \
                .map(self.augment_element, num_parallel_calls=NPROCS)

        # handle perturbation
        if is_perturbed and len(self._data_perturbation)>0:
            dataset = dataset.map(self.perturb_element, num_parallel_calls=NPROCS)
        else:
            dataset = dataset.map(self.duplicate_x_element_if_needed, num_parallel_calls=NPROCS)

        # SHUFFLE, REPEAT and BATCH
        if shuffle:
            LARGE_NUMBER = self._shuffle_buffer
            dataset = dataset.shuffle(LARGE_NUMBER + 1)
        if repeat:
            dataset = dataset.repeat()
            batched_dataset = dataset.batch(batch_size)

            # create iterator to retrieve batches
            iterator = batched_dataset.make_one_shot_iterator()
        else:
            batched_dataset = dataset.batch(batch_size)
            iterator = batched_dataset.make_initializable_iterator()

        return iterator

    def int_to_str_label(self, int_label: int):
        for key, value in LABEL_TO_INT_DICT.items():
            if int_label == value:
                return key
        return None

    @property
    def n_samples_train(self):
        return 95
    # full size: 176 - 52(test) -29(valid) = 95

    @property
    def x_shape_train(self):
        """return the shape of an input sample"""
        return self._x_sample_shape_train

    @property
    def x_shape_eval(self):
        """return the shape of an input sample"""
        return self._x_sample_shape_eval

    @property
    def y_shape(self):
        """return the shape of an output sample"""
        return None

    @property
    def sample_rate(self):
        """return the shape of an output sample"""
        return self._sample_rate
