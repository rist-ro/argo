"""
Module for managing CMB dataset
"""

import numpy as np

import tensorflow as tf

# import gzip
# import pickle

import os
import glob
import timeit

import pdb

import pandas as pd
import json
import operator
from .Dataset import NPROCS, TRAIN, VALIDATION, TEST
from .ImageDataset import ImageDataset
from sklearn.model_selection import train_test_split
# import ipdb as pdb
from scipy.ndimage import gaussian_filter
class LSS(ImageDataset):
    """
    This class manage the dataset CMB a dataset of array generated from
    simulations of CMB anysotropies

    Parameters
    ---------

    params : dict
        dictionary that can contain
        'data_dir' (str)    path (relative or absolute) to the dataset
        'augm_data' (bool or int) if true use also reflected and rotated pictures
        'params_dim' (int) dim of parameter space in the datasets
        'par_distr' (str) lattice  or uniform describes the distribution of the parameter
                      in the range
        'pic_per_run_rot' (int) number of pict per run taken rotating the full map
        'pic_per_run_equator' (int) number of pictures per run at the equator
        'debug_mode' (bool) load only the first 500 pictures



        'id_note' (str)     append information to the id
         +-------------+-----------+-----------+-----------+-------------------------------------------------------------+
         | params Key  | values    | default v |  id short | Short description                                           |
         +=============+===========+===========+===========+=============================================================+
         |             |           |           |           |                                                             |


    """

    default_params = {
        #'dataName': 'CMB',
        "data_dir": '/ssd_data/datasets/CMB', # default value for the dir_name
        # "params_dim": 2,
        # "par_distr": 'uniform',
        # 'pic_per_run_rot': 1,
        # 'pic_per_run_equator':1,
        "normalize_labels" : 1,
        "normalize_images" : 1,
        #"augment_data" : 0,
        #"only_center" : True,
        #'debug_mode' : False,
        #"id": ""
        }

    classification = False

    def __init__(self, params):
        super().__init__(params)

        self._json_filename = "paramslss.json"
        fraction_dataset = params['fraction_dataset']

        # self._csv_filename = "labels_file.csv"
        if fraction_dataset ==100:
            self._csv_filename_Train = 'Train_LSS_data.csv'
        else:
            self._csv_filename_Train = 'Train_LSS_data_{}.csv'.format(fraction_dataset)
        # label_df_Train = pd.read_csv(self._csv_filename_Train, sep="\t")

        self._csv_filename_Test = 'Test_LSS_data.csv'
        # label_df_Test = pd.read_csv(self._csv_filename_Test, sep="\t")

        self._csv_filename_Validation = 'Validation_LSS_data.csv'
        # label_df_Validation = pd.read_csv(self._csv_filename_Validation, sep="\t")

        # self._all_parameter_list = ['h', 'omega_b', 'omega_cdm', 'A_s', 'n_s', 'tau_reio']
        self._parameters_list = params["parameters"]
        self._fraction_dataset = params["fraction_dataset"]

        self._id = self.dataset_id(params)
        self._data_dir = self._params['data_dir']
        self._normalize_labels = self._params['normalize_labels']
        self._normalize_images = self._params['normalize_images']


        # check if the data directory is present
        if not os.path.exists(self._data_dir) and \
            not os.path.isdir(self._data_dir):
            raise Exception('Dataset directory {}'
                            ' not found'.format(self._data_dir)
            )

        #self._debug_mode = params['debug_mode']

        #self._augment_data = params['augm_data']
        #self._only_center = params['only_center']
        json_filename = self._data_dir + "/" + self._json_filename
        with open(json_filename) as f:
            conf_dict = json.loads(f.read())

        picture_size = conf_dict["pic_size"]
        self._x_sample_shape = (picture_size, picture_size, 1)
        #self._y_sample_shape = (self._n_params,)

        #self._var_params = self.conf_dict['params'] # TODO Check that is sorted like self.parameter_list

        # useful for lazy load, however I need to reimplement some methods to make it work
        # self._loaded_from_disk = False

        #self._train_set_x = None
        #self._train_set_y = None
        #self._test_set_x = None
        #self._test_set_y = None

        #self._n_samples_train = None
        #self._n_samples_test = None
        self._labels_min, self._labels_max,\
            self._data_min, self._data_max = self.compute_min_max_data(norm_labels=self._normalize_labels,
                                                                       norm_data=self._normalize_images)

        self._train_set_x, self._train_set_y = self.load_dataset(TRAIN)
        self._validation_set_x, self._validation_set_y = self.load_dataset(VALIDATION)
        self._test_set_x, self._test_set_y = self.load_dataset(TEST)


        self._caching_bool = False
        self._shuffling_cache = None
        # self._shuffling_cache = 3000



    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        # TODO: missing features are  train/test?
        # CMB.check_params_impl(params)

        p_dist_abbr = {'uniform':'U',
                       'lattice':'L'}

        id = 'LSS'
        id += os.path.basename(params['data_dir'])
        id += '-d%'+str(params['fraction_dataset'])

        id += '-n'
        id += '1' if params['normalize_images'] == True else '0'
        id += '1' if params['normalize_labels'] == True else '0'
        #id += '-au1' if self._augment_data == True  else '-au0'
        #id += '-oc1' if self._only_center == True else '-oc0'

        # id += '-pdim' + str(params['params_dim'])
        # id += '-' + p_dist_abbr[params['par_distr']]
        # id += '-pprr' + str(params['pic_per_run_rot'])
        # id += '-ppre' + str(params['pic_per_run_equator'])

        # id note (keep last)
        #if params['id_note']:
        #    id += params['id_note']

        return id


    def read_metadata(self, json_basename, csv_basename):
        #load json file
        json_filename = self._data_dir + "/" + json_basename
        with open(json_filename) as f:
            conf_dict = json.loads(f.read())

        csv_filename = self._data_dir + "/" + csv_basename
        label_df = pd.read_csv(csv_filename, sep="\t")

        return conf_dict, label_df


    def get_output_shapes(self, datasets_tuple):
        image = np.load(self._data_dir + '/' + datasets_tuple[0][0]).astype(np.float32)
        output_shapes = tuple([image.shape + (1,), datasets_tuple[1].shape[1]])

        return output_shapes

    def get_output_types(self, datasets_tuple):
        image = np.load(self._data_dir + '/' + datasets_tuple[0][0]).astype(np.float32)
        output_types = tuple([tf.as_dtype(image.dtype), tf.as_dtype(datasets_tuple[1].dtype)])

        return output_types
    def cropND(self, img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]


    # this function can be used in case I want to lazy load dinamically the files
    # see for instance CMB
    def dataset_map(self, dataset, datasets_tuple):

        output_types = self.get_output_types(datasets_tuple)

        norm_bool = self._normalize_images
        data_min = self._data_min
        data_max = self._data_max


        def load_function(n):
            filename = full_data[n][0]
            label = full_data[n][1]
            image = np.load(self._data_dir + '/' + filename)
            if norm_bool:
                #image = 2*(image-data_min)/(data_max-data_min) - 1

                image=gaussian_filter(image, sigma=1.2)
                image = 5*(2*np.log(image-data_min+1)/np.log(data_max-data_min+1)-1./4.)
                s_random = np.random.uniform(0,1,1)
                if s_random>=0.25:
                    image= np.flipud(image)
                elif 0.5>s_random>0.25:
                    image= np.fliplr(image)
                elif 0.75>s_random>=0.5:
                    image= np.rot90(image,1)
                else:
                    pass
            #image=self.cropND(image, (128,128))
            image = np.expand_dims(image, axis=-1)

            return image.astype(np.float32), label

        full_data = list(zip(*datasets_tuple))

        dataset = dataset.map(
            lambda n: tuple(tf.py_func(load_function,
                                       [n], output_types)
            ), num_parallel_calls=NPROCS)

        return dataset


    def load_dataset(self, dataset_str):
        """
        load the dataset in memory and set in the object

        Args:
            train_test_ratio: (float) percentage of the dataset in train

        """
        json_filename = self._json_filename

        if dataset_str==TRAIN:
            csv_filename = self._csv_filename_Train
        elif dataset_str==VALIDATION:
            csv_filename = self._csv_filename_Validation
        elif dataset_str==TEST:
            csv_filename = self._csv_filename_Test
        else:
            raise Exception("`dataset_str` can be only: train, validation or test.")

        # get info from the metadata
        conf_dict, labels_filter_df = self.read_metadata(json_filename, csv_filename)
        #self._picture_size = self.conf_dict['pic_size']
        n_parameters = len(conf_dict)

        #if self._only_center:
        #    labels_filter_df = labels_filter_df[labels_filter_df['is_center']]

        #if not self._augment_data:
        #    # no rotations
        #    is_no_rotation = labels_filter_df['rot_angle'] == 0
        #    labels_filter_df = labels_filter_df[is_no_rotation]

        #    # no flips
        #    index_no_flip = ~labels_filter_df['flip']
        #    labels_filter_df = labels_filter_df[index_no_flip]


        '''
        if self._debug_mode:
            debug_dimension = 500
            labels_filter_df = labels_filter_df.iloc[:debug_dimension]
        '''

        # dataset_x = labels_filter_df.sample(frac=1)
        df_dataset = labels_filter_df
        num_files = df_dataset.shape[0]

        print('{} are going to be loaded in memory'.format(num_files))

        filename_dataset = df_dataset['filename'].values
        labels_dataset = df_dataset[self._parameters_list].values.astype(np.float32)


        if self._normalize_labels:
            labels_dataset = 2*(labels_dataset-self._labels_min)/(self._labels_max - self._labels_min) - 1.


        #COMMENTED BEFORE REFACTORING
        # dim_train = int(num_files * train_ratio)
        # dim_validation = int(num_files * validation_ratio)
        # dim_test = num_files - dim_train - dim_validation
        #
        # # training
        # df_train = dataset_x.iloc[0 : dim_train]
        # filename_training = df_train['filename'].values
        # labels_training = df_train[['omega_cdm', 'A_s']].values.astype(np.float32)
        # if self._normalize_data:
        #     max_training = np.max(labels_training, axis=0)
        #     min_training = np.min(labels_training, axis=0)
        #     labels_training = (labels_training - min_training)/(max_training - min_training)
        #
        # # testing
        # df_validation = dataset_x.iloc[dim_train : (dim_train+dim_validation)]
        # filename_validation = df_validation['filename'].values
        # labels_validation = df_validation[['omega_cdm','A_s']].values.astype(np.float32)
        # if self._normalize_data:
        #     labels_validation = (labels_validation - min_training)/(max_training - min_training)
        #
        # # testing
        # df_test = dataset_x.iloc[(dim_train+dim_validation):]
        # filename_test = df_test['filename'].values
        # labels_test = df_test[['omega_cdm','A_s']].values.astype(np.float32)
        # if self._normalize_data:
        #     labels_test = (labels_test - min_training)/(max_training - min_training)
        #
        #COMMENTED BEFORE REFACTORING

        return filename_dataset, labels_dataset

    def compute_min_max_data(self, norm_labels=False, norm_data=False):
        all_max = []
        all_min = []
        csv_filename = self._data_dir + '/latin_hypercube.csv'
        label_df = pd.read_csv(csv_filename, sep="\t")

        labels_min = None
        labels_max = None
        data_min = None
        data_max = None

        if norm_data:
            filenames = label_df['filename']

            for filename in filenames:
                patch1 = np.load(self._data_dir + "/" + filename)
                patch1=gaussian_filter(patch1, sigma=1.2)

                all_min.append(np.min(patch1))
                all_max.append(np.max(patch1))

            data_min = np.min(all_min)
            data_max = np.max(all_max)

        if norm_labels:
            labels = label_df[self._parameters_list].values.astype(np.float32)
            labels_min = np.min(labels, axis=0)
            labels_max = np.max(labels, axis=0)

        return labels_min, labels_max, data_min, data_max

    @property
    def labels_min_training(self):
        #if not self._loaded_from_disk:
        #    self.load_dataset_from_disk()
        #
        #if  self._dataset_x_min is None:
        #    raise Exception("No normalization procedure has been done. If you want"
        #                    "dataset_x_min and dataset_x_max, normalize the labels with the"
        #                    "normalize_label flag = True.")
        #else:

        return self._labels_min_training

    @property
    def labels_max_training(self):
        #if not self._loaded_from_disk:
        #    self.load_dataset_from_disk()
        #§
        #if  self._dataset_x_max is None:
        #    raise Exception("No normalization procedure has been done. If you want"
        #                    "dataset_x_min and dataset_x_max, normalize the labels with the"
        #                    "normalize_label flag = True.")
        #else:
        return self._labels_max_training

    @property
    def x_shape_train(self):
        """return the shape of an input sample for the train loop"""
        return self._x_sample_shape

    @property
    def x_shape_eval(self):
        """return the shape of an input sample for the evaluation"""
        return self._x_sample_shape

    # # overriding
    # @property
    # def x_shape_train(self):
    #     return self._train_set_x_shape
    #
    # # overriding
    # @property
    # def x_shape_eval(self):
    #     return self._train_set_x_shape
    #
        #return mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels, mnist.test.images, mnist.test.labels


        #start_time_data = timeit.default_timer()

        #shape_X = (n_files,) + self._x_sample_shape
        #shape_Y = (n_files,) + self._y_sample_shape

        # construct the datasets
        #X = np.empty(shape_X)
        #Y = np.empty(shape_Y)

        #for ix, row in  labels_filter_df.iterrows():
        #    tmp_numpy = np.load(self._data_dir + "/" + row['filename'])
        #
        #    X[ix, :, :, 0] = tmp_numpy
        #    Y[ix] = row[self._var_params]

        # print time for the load
        #step_time = timeit.default_timer()
        #print("time needed to load: ", step_time - start_time_data)

        # shuffle the dataset
        '''
        randomized_dataset_index = np.random.permutation(n_files)
        X = X[randomized_dataset_index]
        Y = Y[randomized_dataset_index]






        dim_train = int(n_files * train_test_ratio)
        self._train_set_x, self._train_set_y = X[:dim_train] , Y[:dim_train]
        self._test_set_x, self._test_set_y = X[dim_train:], Y[dim_train:]
        '''

        #self._n_samples_train = len(self._train_set_y)
        #self._n_samples_test = len(self._test_set_y)

        #self._loaded_from_disk = True


        #stop_time_data = timeit.default_timer()

        #print("plus time to shuffle: ", stop_time_data - step_time, " and total ", stop_time_data -start_time_data)

    '''
    # overidden from Dataset
    def set_labels_attr(self):
        # nothing to be done, since we already have
        # self._n_labels = len(self._labels) in load_dataset()
        pass
    '''



    # def load_dataset_from_disk(self, n_params = None,
    #                            augmented_dataset = False,
    #                            train_test_ratio = 0.7):
    #     """
    #     load the dataset in memory
    #
    #     Args:
    #         n_params: dimension of the Y label
    #         augmented_dataset: (bool) augment the dataset with rotations and
    #                                     mirror
    #         train_test_ratio: (float) percentage of the dataset in train
    #
    #     Returns:
    #         x_train,y_train,
    #         y_test,y_test
    #
    #     """
    #
    #     filename = '.npy' if augmented_dataset else '_ORI.npy' # TODO eliminate
    #
    #
    #     # Read all subdirectories of data_dir.
    #     directories = os.listdir(self._data_dir)
    #
    #     # I suppose that the directory contains only the dataset and
    #     # that they all have the same dimension
    #     def get_values_from_dir_str(params_str):
    #         params_str = params_str.split('_')
    #         params = [float(p) for p in params_str]
    #         return params
    #
    #     # all the directories have to look like the first one if n_params
    #     # is not passed to the function
    #     if not n_params:
    #         check_dir = get_values_from_dir_str(directories[0])
    #         n_params = len(check_dir)
    #
    #     if n_params < 1:
    #         raise FileExistsError('Dataset error, the directories are not '
    #                               'formatted properly')
    #
    #     # scan directories to count the files and construct the dataset
    #     # numpy arrays. TODO If the dimension is too big act accoridingly.
    #
    #     # construct glob path
    #     # e.g 'data_dir/*_*_*/a.npy'
    #     pattern = self._data_dir + ('*_' * (n_params - 1)) + '*/*' + filename
    #
    #     files_paths = glob.glob(pattern)
    #
    #     # the total dataset dimension is
    #     total_number_of_files = len(files_paths)
    #     if total_number_of_files < 1:
    #         raise Exception('No files found')
    #
    #     if self._debug_mode:
    #         debug_dimension = 500
    #         files_paths = files_paths[:debug_dimension]
    #         total_number_of_files = len(files_paths)
    #
    #     print('{} are going to be laoded in memory'.format(total_number_of_files))
    #     start_time_data = timeit.default_timer()
    #     file_to_test = np.load(files_paths[0])
    #
    #     picture_size = self.conf_dict
    #     single_file_shape =
    #     shape_X = list(single_file_shape)
    #     shape_X.insert(0,total_number_of_files)
    #     #shape_X =  (total_number_of_files,  ) +single_file_shape
    #
    #
    #     #construct the datasets
    #     X = np.empty(tuple(shape_X))
    #     Y = np.empty((total_number_of_files,n_params))
    #
    #     # populate
    #     for ix,fl in enumerate(files_paths):
    #         tmp_numpy = np.load(fl)
    #
    #         X[ix,:,:,0] = tmp_numpy
    #
    #         dname = fl.replace(self._data_dir,'').split('/')
    #         params = get_values_from_dir_str(dname[0])
    #
    #         Y[ix] = params
    #
    #
    #     # add channel to the shape that is a single channel
    #     new_shape_with_channel = X.shape + (1,)
    #     X = X.reshape(new_shape_with_channel)
    #     # print(X.shape)
    #     randomized_dataset_index = list(range(total_number_of_files))
    #     np.random.shuffle(randomized_dataset_index)
    #
    #     # print(randomized_dataset_index)
    #     X=X[randomized_dataset_index]
    #     Y=Y[randomized_dataset_index]
    #
    #
    #     dim_train = int(total_number_of_files * train_test_ratio)
    #
    #
    #
    #     X_train, Y_train = X[:dim_train] , Y[:dim_train]
    #     X_test, Y_test = X[dim_train:], Y[dim_train:]
    #
    #     stop_time_data = timeit.default_timer()
    #     print("time needed to load: ", stop_time_data - start_time_data)
    #
    #
    #     return X_train, Y_train, X_test, Y_test









    # @staticmethod
    # def sub_sample(data_set_x, data_set_y, subsampling):
    #     """
    #     return a value every "subsampling"
    #
    #     :param data_set_x
    #     :param data_set_y
    #     :param subsampling: integer < dim(data_set)
    #     :return: dataset_x, dataset_y
    #     """
    #
    #     len_train = len(data_set_x)
    #     reshuf_index_train = np.random.permutation(len_train)
    #     new_len_train = int(len_train / subsampling)
    #
    #     data_set_x = data_set_x[reshuf_index_train[:new_len_train]]
    #     data_set_y = data_set_y[reshuf_index_train[:new_len_train]]
    #
    #     return data_set_x, data_set_y

    # @staticmethod
    # def class_filter(data_set_x, data_set_y, classes, position_label):
    #     """
    #     return the dataset with labels in the list classes
    #
    #     :param data_set_x: data
    #     :param data_set_y: labels
    #     :param classes:    list of classes
    #     :param position_label:  list of classes
    #     :return: (dataset_x, dataset_y) with filtered elemnts not in classes
    #     """
    #
    #     ix_mtch_class_train = np.in1d(data_set_y, classes)
    #     data_set_x = data_set_x[ix_mtch_class_train]
    #     data_set_y = data_set_y[ix_mtch_class_train]
    #     if position_label:
    #
    #         def replace_with_position(label_set, classes):
    #             label_set_new = np.copy(label_set)
    #             for ix, class_ in enumerate(classes): label_set_new[label_set == class_] = ix
    #             return label_set_new
    #
    #         data_set_y = replace_with_position(data_set_y, classes)
    #
    #     return data_set_x, data_set_y

    '''
    def get_data_dict(self):

        if not self._binary_input or (self.params['binary'] and not self.params['stochastic']):
            ds["train_set_y"] = self._train_set_y
            ds["test_set_y"] = self._test_set_y

        return ds
    '''

    '''
    @property
    def x_shape(self):
        """return the shape of an input sample"""
        return self._x_sample_shape


    @property
    def y_shape(self):
        """return the shape of an output sample"""
        return self._y_sample_shape

    @property
    def input_size(self):
        """ DEPRECATED"""
        print("WARNING in input_size ! shape should be used")
        return self._x_sample_shape


    @property
    def output_size(self):
        """ DEPRECATED"""
        print("WARNING in output_size! shape should be used")
        return self._y_sample_shape

    @property
    def size_images(self):
        return self._x_sample_shape[:-1] #remove the channels

    @property
    def n_samples_train(self):
        if not self._loaded_from_disk:
            self.load_dataset_from_disk()

        return self._n_samples_train

    @property
    def n_samples_test(self):
        if not self._loaded_from_disk:
            self.load_dataset_from_disk()

        return  self._n_samples_test


    @property
    def train_set_x(self):
        if not self._loaded_from_disk:
            self.load_dataset_from_disk()

        return self._train_set_x

    @property
    def train_set_y(self):
        if not self._loaded_from_disk:
            self.load_dataset_from_disk()

        if self._train_set_y is None:
            raise Exception("Labels not available")
        else:
            return self._train_set_y

    '''

    '''
    @property
    def test_set_x(self):
        if not self._loaded_from_disk:
            self.load_dataset_from_disk()

        if self._test_set_x is None:
            raise Exception("Test set not available")
        else:
            return self._test_set_x

    @property
    def test_set_y(self):
        if not self._loaded_from_disk:
            self.load_dataset_from_disk()

        if self._test_set_x is None:
            raise Exception("Test set not available")
        else:
            if self._test_set_y is None:
                raise Exception("Labels not available")
            else:
                return self._test_set_y
    '''







# @staticmethod
    # def generator_id(data_generator_parameters):
    #     """
    #     create an id for the generator process
    #
    #     Args:
    #         data_generator_parameters: (dict) parameters
    #
    #     Returns:
    #         (str) the id
    #
    #     """
    #
    #     data_generator_parameters = {
    #         "data_dir": '/ssd_data/CMB/',
    #         # "n_runs": 10,
    #         # "n_parameter_samples": 20,
    #         # "params": { 'h' ,'omega_b' ,'omega_cdm','A_s','n_s','tau_reio'},
    #         "params": {'omega_cdm', 'A_s'},
    #         "par_distr": 'uniform',
    #         'pic_per_run_rot': 0,
    #         'pic_per_run_equator': 1,
    #         'pic_per_run_latitude': 1,
    #         "augm_data": 0,
    #         'patch_size': 10,
    #         'pic_size': 224,
    #         #'parallel': False
    #     }
    #
    #
    #     parameter_list = ['h', 'omega_b', 'omega_cdm', 'A_s', 'n_s',
    #                       'tau_reio']
    #
    #     all_parameters_list = parameter_list.copy()
    #
    #     # remove from the   cls.parameter_list the elements that do not appears
    #     # in data_generator_parameters['params']
    #
    #     parameters_sampled = data_generator_parameters['params']
    #     all_parameters_set = set(parameter_list)
    #
    #     not_used_parameters = all_parameters_set.difference(parameters_sampled)
    #
    #     for i in not_used_parameters:
    #         all_parameters_list.remove(i)
    #
    #     # create the id
    #
    #     id = "_".join(all_parameters_list)
    #     id += ("-" + data_generator_parameters['par_distr'])
    #     id += ("-pppr_rot" + str(data_generator_parameters['pic_per_run_rot']))
    #     id += ("-pppr_eq" + str(
    #         data_generator_parameters['pic_per_run_equator']))
    #     id += ("-aug" + str(data_generator_parameters['augm_data']))
    #     id += ("+patch_s" + str(data_generator_parameters['patch_size']))
    #     id += ("+pic_s" + str(data_generator_parameters['pic_size']))
    #
    #     return id
