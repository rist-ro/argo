"""
Module for managing the Sinusoids dataset
"""
from functools import partial

import numpy as np
import tensorflow as tf

from datasets.AudioDataset import AudioDataset
from datasets.Dataset import TRAIN, VALIDATION, TEST

from scipy import signal

DATASET_SIZE = 1000
SAMPLE_LENGTH = 1280
TYPE_CLEAN = "clean"
TYPE_FADE = "fade"
TYPE_FADE_IN = "fadein"
TYPE_FADE_OUT = "fadeout"
TYPE_SWITCH = "switch"
TYPE_MIX = "mix"
TYPE_MIX_CONDITIONED = 'mix_conditioned'
TYPE_SAW_TOOTH = 'saw_tooth'
TYPES = [TYPE_MIX, TYPE_CLEAN, TYPE_FADE, TYPE_FADE_IN, TYPE_FADE_OUT, TYPE_SWITCH, TYPE_SAW_TOOTH,
         TYPE_MIX_CONDITIONED]
LABEL_TO_INT_DICT = {
    TYPE_CLEAN:     0,
    TYPE_FADE_IN:   1,
    TYPE_FADE_OUT:  2,
    TYPE_SAW_TOOTH: 3}


class Sinusoids(AudioDataset):
    """
    This class manage the dataset Sinusoids, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default
    one. It then return a unique id identifier

    """
    default_params = {
        'crop_length':    512,
        'shuffle_buffer': 40000,
    }

    classification = False  # true if

    implemented_params_keys = ['dataName']  # all the admitted keys

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        self._binary_input = True

        # Width and height of each image.
        self._sample_lenght = SAMPLE_LENGTH * 10  # 128 padded 0s  # used to be 88384 for sample rate 44100
        self._sample_rate = 44100
        self._shuffle_buffer = self._params['shuffle_buffer']
        self._type = self._params['type']

        self._train_set_x, self._train_set_y, \
        self._validation_set_x, self._validation_set_y, \
        self._test_set_x, self._test_set_y = self.load_data(self._type)

        self._x_sample_shape_train = [self._crop_length_train, 1]
        self._x_sample_shape_eval = [self._sample_lenght, 1]
        self._y_sample_shape = None  # (self._n_labels,)

        self._n_labels = len(LABEL_TO_INT_DICT)

    def get_dataset_iterator(self, batch_size, dataset_str, shuffle, repeat, augment, perturb):
        is_perturbed = False
        datasets_tuple = None

        # create Dataset objects
        if dataset_str == TRAIN:
            datasets_tuple = (self.train_set_x,)
            if hasattr(self, "perturbed_train_set_x") and self.perturbed_train_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_train_set_x,)
                is_perturbed = True
            if self._train_set_y is not None:
                datasets_tuple = datasets_tuple + (self.train_set_y,)

        elif dataset_str == VALIDATION:
            datasets_tuple = (self.validation_set_x,)
            if hasattr(self, "perturbed_validation_set_x") and self.perturbed_validation_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_validation_set_x,)
                is_perturbed = True
            if self._validation_set_y is not None:
                datasets_tuple = datasets_tuple + (self.validation_set_y,)

        elif dataset_str == TEST:
            datasets_tuple = (self.test_set_x,)
            if hasattr(self, "perturbed_test_set_x") and self.perturbed_test_set_x is not None:
                datasets_tuple = datasets_tuple + (self.perturbed_test_set_x,)
                is_perturbed = True
            if self._test_set_y is not None:
                datasets_tuple = datasets_tuple + (self.test_set_y,)

        else:
            raise Exception("dataset not recognized (accepted values are: train, validation and test)")

        NPROCS = 20
        n_samples = datasets_tuple[0].shape[0]
        dataset = tf.data.Dataset.range(n_samples)

        output_shapes = self.get_output_shapes(datasets_tuple)

        dataset = self.dataset_map(dataset, datasets_tuple)

        def _set_shapes(*nodes):
            for node, outshape in zip(nodes, output_shapes):
                node.set_shape(outshape)
            return nodes

        dataset = dataset.map(_set_shapes, num_parallel_calls=NPROCS)

        # caching before shuffling and batching for super cow speed
        dataset = dataset.cache()

        # AUGMENT DATA (AUGMENT IF NEEDED)
        if augment:
            dataset = dataset.map(partial(self._crop_element, is_perturbed, self._crop_length_train),
                                  num_parallel_calls=NPROCS) \
                .map(self.augment_element, num_parallel_calls=NPROCS)
        if dataset_str == TEST:
            dataset = dataset.map(partial(self._crop_element, is_perturbed, self._crop_length_train),
                                  num_parallel_calls=NPROCS) \
                .map(self.augment_element, num_parallel_calls=NPROCS)

        # handle perturbation
        if is_perturbed and len(self._data_perturbation) > 0:
            dataset = dataset.map(self.perturb_element, num_parallel_calls=NPROCS)
        else:
            dataset = dataset.map(self.duplicate_x_element_if_needed, num_parallel_calls=NPROCS)

        # SHUFFLE, REPEAT and BATCH
        # we shuffle the data and sample repeatedly batches for training
        if shuffle:
            if self._shuffling_cache is None:
                shuffling_cache = n_samples + 1
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

        return iterator

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        Sinusoids.check_params_impl(params)

        _id = 'SIN'

        if 'type' not in params.keys():
            raise ValueError("Type has to be defined for Sinusoids ")
        if params['type'] in TYPES:
            _id += '-tp' + params['type']
        else:
            raise ValueError("Type not recognized for Sinusoids, available types: {}".format(TYPES))

        crop_length = params['crop_length']
        if crop_length != Sinusoids.default_params['crop_length']:
            _id += '-cl' + str(crop_length)
        return _id

    @staticmethod
    def generate_dataset(nr_of_samples, dataset_type):
        if dataset_type == TYPE_MIX_CONDITIONED:
            return Sinusoids.generate_mix_with_conditioning(nr_of_samples)

        # The dataset is defined in the Tutorial on Helmholtz Machines by Kevin G Kirby
        dataset_x = []
        dataset_y = []

        def get_sin(ampl=1.0, freq=1.0, offset=0.0, length=100):
            x = np.arange(0.0, length, 0.1) / freq
            sin = np.sin(offset + x) * ampl
            return sin

        def get_sin_from_sample(sample):
            a = sample % 20 / 20 + 0.1
            a = 1  # Signal compression
            magnification = 0.55  # previously 5 , the less the wav will have higher frequency
            f = (int(sample / 20) % 20 + 1) * magnification + 1
            # print(a, f, o)
            return get_sin(ampl=a, freq=f, offset=0, length=SAMPLE_LENGTH), a, f

        for i in range(nr_of_samples):
            if dataset_type == TYPE_CLEAN:
                s = np.random.randint(400)
                sin, a, f = get_sin_from_sample(s)
            elif dataset_type == TYPE_SAW_TOOTH:
                num_tooths = np.random.randint(10, 35)
                t = np.linspace(0, 1, SAMPLE_LENGTH * 10)
                sin = signal.sawtooth(2 * np.pi * num_tooths * t - np.pi)
                a, f = 0.0, 0.0
            elif dataset_type == TYPE_FADE_OUT:
                s = np.random.randint(400)
                sin, a, f = get_sin_from_sample(s)
                fader_f = np.linspace(0., 1., len(sin))
                sin = sin * fader_f
            elif dataset_type == TYPE_FADE_IN:
                s = np.random.randint(400)
                sin, a, f = get_sin_from_sample(s)
                fader_b = np.linspace(1., 0., len(sin))
                sin = sin * fader_b
            elif dataset_type == TYPE_MIX:
                s1 = np.random.randint(400)
                s2 = np.random.randint(400)
                sin1, a1, f1 = get_sin_from_sample(s1)
                sin2, a2, f2 = get_sin_from_sample(s2)
                sin = sin1 * sin2
                a, f = (0.0, 0.0)
            elif dataset_type == TYPE_FADE:
                s1 = np.random.randint(400)
                s2 = np.random.randint(400)
                sin1, a1, f1 = get_sin_from_sample(s1)
                sin2, a2, f2 = get_sin_from_sample(s2)
                fader_f = np.linspace(0., 1., len(sin1))
                fader_b = np.linspace(1., 0., len(sin1))
                sin = sin1 * fader_f + sin2 * fader_b
                a, f = (0.0, 0.0)
            elif dataset_type == TYPE_SWITCH:
                s1 = np.random.randint(400)
                s2 = np.random.randint(400)
                sin1, a1, f1 = get_sin_from_sample(s1)
                sin2, a2, f2 = get_sin_from_sample(s2)
                halfway = int(len(sin1) / 2)
                sin = [*sin1[:halfway], *sin2[halfway:]]
                a, f = (0.0, 0.0)

            else:
                raise ValueError("Type not recognized for Sinusoids, available types: {}".format(TYPES))
            dataset_x += [sin]
            dataset_y += [[a, f]]

        len_sample = len(dataset_x[0])
        dataset_x = np.asarray(dataset_x)
        dataset_y = np.asarray(dataset_y)

        # make them the right length pictures
        return dataset_x.reshape((nr_of_samples, len_sample, 1)), dataset_y

    @staticmethod
    def generate_mix_with_conditioning(nr_samples):
        # generate multiple types of sin waves: fade_in, fade_out, clean, saw_tooth
        types = [TYPE_CLEAN, TYPE_FADE_IN, TYPE_FADE_OUT, TYPE_SAW_TOOTH]
        samples_per_type = nr_samples // len(types)
        dataset_x = []
        dataset_y = []

        for type_id, type in enumerate(types):
            one_hot = np.zeros((samples_per_type, len(types)), dtype='uint8')
            one_hot[:, type_id] = 1
            waves, _ = Sinusoids.generate_dataset(samples_per_type, type)
            dataset_x.append(waves)

            # if dataset_y is None:
            #     dataset_y = one_hot
            # else:
            # dataset_y = np.append(dataset_y, one_hot, axis=0)
            dataset_y = np.append(dataset_y, [type_id]*samples_per_type, axis=0)

        return np.concatenate(dataset_x, axis=0), dataset_y.astype(dtype=np.int32)

    @staticmethod
    def load_data(dataset_type=TYPE_CLEAN):
        # see https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
        sinusoids, features = Sinusoids.generate_dataset(DATASET_SIZE, dataset_type=dataset_type)

        perm = np.random.permutation(DATASET_SIZE)

        data_x = sinusoids[perm].astype(np.float32)
        # data_y = features[perm].astype(np.float32)
        data_y = features[perm] #.astype(np.float32)

        partition = int(DATASET_SIZE / 6)
        # extra is ignored, since the images are too simple
        return data_x[:DATASET_SIZE - partition], \
               data_y[:DATASET_SIZE - partition], \
               data_x[DATASET_SIZE - partition:], \
               data_y[DATASET_SIZE - partition:], \
               data_x[DATASET_SIZE - partition:], \
               data_y[DATASET_SIZE - partition:]

    @property
    def n_samples_train(self):
        return DATASET_SIZE  # 39737 used to be with 44140

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

    def int_to_str_label(self, int_label: int):
        for key, value in LABEL_TO_INT_DICT.items():
            if int_label == value:
                return key
        return None