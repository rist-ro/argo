"""
Module for managing WavFromGeneralMidi dataset
"""

from functools import partial

import numpy as np
import tensorflow as tf

from datasets.AudioDataset import AudioDataset
from .Dataset import TRAIN, VALIDATION, TEST

ALL_LABELS = ["instrument_family", "instrument_name", "pitch", "vol", "audio"]

PITCH = "pitch"
VOL = "vol"
AUDIO = "audio"
INSTR_NAME = "instrument_name"
INSTR_FAMILY = "instrument_family"

ALL_LABELS_IMPLEMENTED = [PITCH,
                          VOL,
                          AUDIO,
                          INSTR_NAME,
                          INSTR_FAMILY]


class WavFromGeneralMidi(AudioDataset):
    """
    This class manage the dataset WavFromGeneralMidi, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """

    default_params = {
        'test_and_validation': 8515,
        'data_dir':            "/ssd_data/datasets/wav-from-general-midi",
        'crop_length':         6144,
        'shuffle_buffer':      60000,
        'label':               PITCH
    }

    classification = False  # true if

    def __init__(self, params):
        super().__init__(WavFromGeneralMidi.process_params(params))

        self._id = self.dataset_id(self._params)

        self._data_dir = self._params['data_dir']

        # Width and height of each image.
        self._sample_lenght = 32128 + 128  # 128 padded 0s  # used to be 88384 for sample rate 44100
        self._sample_rate = 16000
        self._shuffle_buffer = self._params['shuffle_buffer']
        self._label = (self._params['label'] if self._params['label'] in ALL_LABELS_IMPLEMENTED else None)

        # self._n_labels = len(self._params['features']) if 'features' in self._params else 0
        self._x_sample_shape_train = [self._crop_length_train, 1]
        self._x_sample_shape_eval = [self._sample_lenght, 1]
        self._y_sample_shape = None  # (self._n_labels,)

        # self._var_labels = self.get_var_labels(self._params['features']) if 'features' in self._params else []

        # self._train_set_x, self._validation_set_x, self._test_set_x, self._train_set_y, self._validation_set_y, self._test_set_y = [None] * 6

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        WavFromGeneralMidi.check_params_impl(params)

        _id = 'WavFromGeneralMidi'

        # TODO I know this is bad but if id is a static method and not an object method I have no idea how to handle this better,
        # TODO soon we should refactor the id to be multilayer of abstraction as in Networks and Models.
        if params['shuffle_buffer'] != WavFromGeneralMidi.default_params['shuffle_buffer']:
            _id += '-sh%.2e' % params['shuffle_buffer']

        return _id

    def get_var_labels(self, param):
        var_params = np.intersect1d(ALL_LABELS, param)

        assert len(var_params) == len(param), "It seems like you might have a mistake in you label name"
        return var_params

    @staticmethod
    def _parse_function(example_proto, y_label=None):
        features = {
            INSTR_FAMILY: tf.FixedLenFeature([], dtype=tf.string),
            INSTR_NAME:   tf.FixedLenFeature([], dtype=tf.string),
            PITCH:        tf.FixedLenFeature([1], dtype=tf.int64),
            VOL:          tf.FixedLenFeature([1], dtype=tf.int64),
            AUDIO:        tf.FixedLenFeature([32128], dtype=tf.float32),
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # add channel info for standardized input

        # We have to add 128 0's for the length to be devisible by 512, which is a common hop_size for this network
        audio = tf.expand_dims(tf.concat([parsed_features[AUDIO], [0] * 128], axis=0), axis=-1)
        # audio = parsed_features["audio"]
        if y_label:
            return (audio, parsed_features[y_label],)
        else:
            return (audio,)

    def get_dataset_iterator(self, batch_size, dataset_str, shuffle, repeat, augment):

        is_perturbed = False
        filename = ""

        # create Dataset objects using the data previously downloaded
        if dataset_str == TRAIN:
            filename = self._data_dir + "/train-wav16-from-general-midi-sounds-float-shuffled.tfrecords"

        elif dataset_str == VALIDATION:
            filename = self._data_dir + "/validation-wav16-from-general-midi-sounds-float-shuffled.tfrecords"

        elif dataset_str == TEST:
            filename = self._data_dir + "/test-wav16-from-general-midi-sounds-float-shuffled.tfrecords"


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

        # PREPROCESS DATA (CROP IF NEEDED, this is needed only for the train loop)
        if dataset_str == TRAIN:
            dataset = dataset.map(partial(self._crop_element, is_perturbed), num_parallel_calls=NPROCS)

        # caching before shuffling and batching for super cow speed
        dataset = dataset.cache()

        # PREPROCESS DATA (AUGMENT IF NEEDED)
        if augment:
            dataset = dataset.map(partial(self._preprocess_element, is_perturbed), num_parallel_calls=NPROCS)

        # SHUFFLE, REPEAT and BATCH
        # we shuffle the data and sample repeatedly batches for the training loop
        if shuffle:
            if self._shuffling_cache is None:
                shuffling_cache = self._shuffle_buffer + 1
            else:
                shuffling_cache = self._shuffling_cache

            dataset = dataset.shuffle(shuffling_cache)

        if repeat:
            dataset = dataset.repeat()
            batched_dataset = dataset.batch(batch_size)
            # create iterator to retrieve batches
            iterator = batched_dataset.make_one_shot_iterator()
        else:
            batched_dataset = dataset.batch(batch_size)
            iterator = batched_dataset.make_initializable_iterator()

        return iterator, is_perturbed

    @property
    def n_samples_train(self):
        return 39820  # 39737 used to be with 44140

    # full size: 56850 - 8515(test) -8515(train) = 39820

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
