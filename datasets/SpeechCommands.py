"""
Module for managing WavFromGeneralMidi dataset
"""

from functools import partial

import numpy as np
import tensorflow as tf

from datasets.AudioDataset import AudioDataset
from .Dataset import TRAIN, VALIDATION, TEST

SPEAKER = "speaker"
WORD = "word"
HASH = "hash"
FILENAME = "filename"
AUDIO = "audio"

ALL_LABELS = [SPEAKER, WORD, HASH, FILENAME, AUDIO]

ALL_LABELS_IMPLEMENTED = ALL_LABELS


class SpeechCommands(AudioDataset):
    """
    This class manage the dataset SpeechCommands, properties of the datasets are uniquely determined
    by the params dictionary

    https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

    """

    default_params = {
        'test_and_validation': 8515,
        'data_dir':            "/ssd_data/datasets/SpeechCommands",
        'shuffle_buffer':      60000,
        'label':               WORD
    }

    classification = False  # true if

    def __init__(self, params):
        super().__init__(SpeechCommands.process_params(params))

        self._id = self.dataset_id(self._params)

        self._data_dir = self._params['data_dir']

        # Width and height of each image.
        self._sample_lenght = 16000 + 384  # 384 padded 0s sometimes
        self._sample_rate = 16000
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

        SpeechCommands.check_params_impl(params)

        _id = 'SpeechC'

        if params['shuffle_buffer'] != SpeechCommands.default_params['shuffle_buffer']:
            _id += '-sh%.2e' % params['shuffle_buffer']

        return _id

    def get_var_labels(self, param):
        var_params = np.intersect1d(ALL_LABELS, param)

        assert len(var_params) == len(param), "It seems like you might have a mistake in you label name"
        return var_params

    @staticmethod
    def _parse_function(example_proto, y_label=None):
        features = {
            SPEAKER:  tf.FixedLenFeature([1], dtype=tf.int64),
            WORD:     tf.FixedLenFeature([], dtype=tf.string),
            HASH:     tf.FixedLenFeature([], dtype=tf.string),
            FILENAME: tf.FixedLenFeature([], dtype=tf.string),
            AUDIO:    tf.VarLenFeature(dtype=tf.float32),
            # AUDIO: tf.FixedLenFeature([111468], dtype=tf.float32),
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # add channel info for standardized input
        parsed_features = tf.parse_single_example(example_proto, features)

        # Expand dims for channels
        audio = tf.expand_dims(tf.concat(parsed_features[AUDIO].values, axis=0), axis=-1)
        if y_label:
            return (audio, parsed_features[y_label],)
        else:
            return (audio,)


    def get_dataset_iterator(self, batch_size, dataset_str, shuffle, repeat, augment):

        is_perturbed = False
        filename = ""

        # create Dataset objects using the data previously downloaded
        if dataset_str == TRAIN:
            filename = self._data_dir + "/train-sc-float-shuffled.tfrecords"

        elif dataset_str == VALIDATION:
            filename = self._data_dir + "/validation-sc-float-shuffled.tfrecords"

        elif dataset_str == TEST:
            filename = self._data_dir + "/test-sc-float-shuffled.tfrecords"

        # TODO: BIIIG TODO I forgot to add the _backgroundnoises_ to the samples. That should be a perturbation step

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
            dataset = dataset.map(partial(self._crop_element, is_perturbed), num_parallel_calls=NPROCS) \
                .map(partial(self._preprocess_element, is_perturbed), num_parallel_calls=NPROCS)

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

        return iterator, is_perturbed

    @property
    def n_samples_train(self):
        return 84843

    # full size: 105829 - 9981(vals) - 11005(test) = 84843

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
