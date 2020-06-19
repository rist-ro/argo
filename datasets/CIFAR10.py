"""
Module for managing CIFAR10 dataset
"""

########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os
import sys
import tarfile

from .utils import normalize, min_max_data_np

import os.path
import urllib.request

from .ImageDataset import ImageDataset    ## TODO  Check wiythout point

import pdb

class CIFAR10(ImageDataset):
    """
    This class manage the dataset CIFAR10, properties of the datasets are uniquely determined
    by the params dictionary

    It compares the parameters and complete them with the default one. It then return a unique id identifier

    Parameters
    ---------

    params : dict
        dictionary that can contain
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | params Key  | values    | default v |  id short | Short description                                           |
         +=============+===========+===========+===========+=============================================================+
         |   binary    |  0,1      |  0        | "-c", "-d"| load continuous or binary CIFAR10                           |
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         |  stochastic |  0,1      |  0        |   "-st"   | sample using Bernoulli from the continuous CIFAR10 after every|
         |             |           |           |           | epoch during training, see IWAE and LVAE papers that claim  |
         |             |           |           |           | this techniques reduces overfitting; this function only     |
         |             |           |           |           | loads continuous CIFAR10 to be used later                   |
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

        self._binary_input = 0

        ########################################################################
        # Various constants for the size of the images.
        # Use these constants in your own program.

        # Width and height of each image.
        self._img_size = 32

        # Number of channels in each image, 3 channels: Red, Green, Blue.
        self._num_channels = 3

        # Length of an image when flattened to a 1-dim array.
        self._img_size_flat = self._img_size * self._img_size * self._num_channels

        # Number of classes.
        self._num_classes = 10

        ########################################################################
        # Various constants used to allocate arrays of the correct size.

        # Number of files for the training-set.
        self._num_files_train = 5

        # Number of images for each batch-file in the training-set.
        self._images_per_file = 10000

        # Total number of images in the training-set.
        # This is used to pre-allocate arrays for efficiency.
        self._num_images_train = self._num_files_train * self._images_per_file

        self._params['binary'] = 0

        self._data_dir = "/ssd_data/datasets/CIFAR10"
        self._data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self._data_extract = self._data_dir + "/cifar-10-batches-py"

        self.maybe_download_and_extract()

        imgs_train, cls_train = self.load_training_data()
        imgs_validation, cls_validation, imgs_test, cls_test = self.load_validation_test_data()

        self._train_set_x = self.patch_data(imgs_train, self._img_size)
        self._validation_set_x = self.patch_data(imgs_validation, self._img_size)
        self._test_set_x = self.patch_data(imgs_test, self._img_size)
        self._train_set_y = cls_train
        self._validation_set_y = cls_validation
        self._test_set_y = cls_test

        #import pdb;pdb.set_trace()

        #normalize data
        all_min, all_max = min_max_data_np([self._train_set_x, self._validation_set_x, self._test_set_x])
        self._train_set_x = normalize(self._train_set_x, all_min, all_max)
        self._validation_set_x = normalize(self._validation_set_x, all_min, all_max)
        self._test_set_x = normalize(self._test_set_x, all_min, all_max)

        # filter classes
        if self._params['classes']:
            position_label = self._params['position_label']
            self._train_set_x, self._train_set_y  = \
                self.class_filter(self._train_set_x, self._train_set_y, self._params['classes'], position_label)
            self._validation_set_x, self._validation_set_y  = \
                self.class_filter(self._validation_set_x, self._validation_set_y, self._params['classes'], position_label)
            self._test_set_x, self._test_set_y = \
                self.class_filter(self._test_set_x, self._test_set_y, self._params['classes'],  position_label)

        # choose a subset
        if self._params['subsampling']:
            self._train_set_x, self._train_set_y = \
                self.sub_sample(self._train_set_x, self._train_set_y, self._params['subsampling'])
            self._validation_set_x, self._validation_set_y = \
                self.sub_sample(self._validation_set_x, self._validation_set_y, self._params['subsampling'])
            self._test_set_x, self._test_set_y = \
                self.sub_sample(self._test_set_x, self._test_set_y, self._params['subsampling'])

        #clip
        clip_low = self._params['clip_low']
        clip_high = self._params['clip_high']
        if (clip_low is not None) or (clip_high is not None):
            m = clip_low if clip_low is not None else 0
            M = clip_high if clip_high is not None else 1
            self._train_set_x = np.clip(self._train_set_x, a_min=m, a_max=M)
            self._validation_set_x = np.clip(self._validation_set_x, a_min=m, a_max=M)
            self._test_set_x = np.clip(self._test_set_x, a_min=m, a_max=M)

    implemented_params_keys = ['dataName', 'binary', 'stochastic', 'classes',
                              'position_label', 'subsampling', 'clip_high', 'clip_low',
                              'data_dir', 'id_note']  # all the admitted keys

    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?
        CIFAR10.check_params_impl(params)

        id = 'CIFAR10'

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
                        "classes contains labels not present in CIFAR10"
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
    @property
    def input_size(self):
        return self._img_size_flat

    @property
    def output_size(self):
        return self._num_classes if self._params['classes'] == [] else len(self._params['classes'])


    @property
    def color_images(self):
        return 1

    @property
    def size_images(self):
        return (self._img_size, self._img_size)
    '''

    def _get_file_path(self, filename=""):
        """
        Return the full path of a data-file for the data-set.
        If filename=="" then return the directory of the files.
        """

        return os.path.join(self._data_dir, "cifar-10-batches-py/", filename)

    def _unpickle(self, filename):
        """
        Unpickle the given file and return the data.
        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        file_path = self._get_file_path(filename)

        print("Loading data: " + file_path)

        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

        return data

    def _convert_images(self, raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=np.float32) / 255.0
        # pdb.set_trace()

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self._num_channels, self._img_size, self._img_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images

    def _load_data(self, filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = self._unpickle(filename)

        # Get the raw images.
        raw_images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        cls = np.array(data[b'labels'])
        #pdb.set_trace()

        # Convert the images.
        images = self._convert_images(raw_images)

        return images, cls


    ########################################################################
    # Public functions that you may call to download the data-set from
    # the internet and load the data into memory.
    ########################################################################

    # see https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py
    def maybe_download_and_extract(self):
        """
        Download and extract the CIFAR-10 data-set if it doesn't already exist
        in data_path (set this variable first to the desired path).
        """

        dest_directory = self._data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self._data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(self._data_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

        #filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(self._data_extract):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def load_class_names(self):
        """
        Load the names for the classes in the CIFAR-10 data-set.
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """

        # Load the class-names from the pickled file.
        raw = self._unpickle(filename="batches.meta")[b'label_names']

        # Convert from binary strings.
        names = [x.decode('utf-8') for x in raw]

        return names

    def load_training_data(self):
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = np.zeros(shape=[self._num_images_train, self._img_size, self._img_size, self._num_channels], dtype=np.float32)
        cls = np.zeros(shape=[self._num_images_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(self._num_files_train):
            # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = self._load_data(filename="data_batch_" + str(i + 1))

            # Number of images in this batch.
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            # Store the images into the array.
            images[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            cls[begin:end] = cls_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        return images, cls

    def load_validation_test_data(self):
        """
        Load all the test-data for the CIFAR-10 data-set.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        images, cls = self._load_data(filename="test_batch")

        #return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

        rnd_seed = 42
        n_images = len(images)
        np.random.seed(rnd_seed)
        perm = np.random.permutation(n_images)
        images = images[perm]
        cls = cls[perm]
        percentage_train = 0.5
        k = int(n_images*percentage_train)

        #return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)
        return images[:k], cls[:k], images[k:], cls[k:]


    def patch_data(self, data, patch_size):
        patch_data = np.zeros(shape=(data.shape[0], patch_size, patch_size, self._num_channels), dtype=np.float32)
        patch_index = np.random.randint(self._img_size-patch_size+1, size=(data.shape[0], 2))
        for i in range(data.shape[0]):
            patch_data[i] = data[i, patch_index[i, 0]:patch_index[i, 0]+patch_size,
                             patch_index[i, 1]:patch_index[i, 1]+patch_size, :]
        return patch_data #.reshape(data.shape[0], patch_size*patch_size*self._num_channels)
