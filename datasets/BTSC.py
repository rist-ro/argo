"""
Module for managing Belgium Traffic Signs Classification dataset
"""

from .ImageDataset import ImageDataset

import h5py
import sklearn
import sklearn.model_selection
import os
import pandas as pd
import skimage.data
import skimage.transform
from skimage import io, color, exposure, transform
import glob
import urllib
from .utils import normalize, min_max_data_np

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

import pdb

NUM_CLASSES = 62
IMG_SIZE = 32


class BTSC(ImageDataset):
    """
    This class manage the dataset BTSC, properties of the datasets are uniquely determined
    by the params dictionary

    """

    default_params = {
        # "something" : param
        }

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        default_data_dir = '/ssd_data/datasets/BTSC'

        self.data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir

        self._train_set_x, self._train_set_y, \
            self._validation_set_x,  self._validation_set_y, \
            self._test_set_x,  self._test_set_y = self.load_data(self.data_dir)


    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        BTSC.check_params_impl(params)

        id = 'BTSC'

        return id

    @staticmethod
    def load_data(btsc_dir):

        try:
            os.stat(btsc_dir)
        except:
            os.mkdir(btsc_dir)

        X_train, Y_train, X_val, Y_val = load_train(os.path.join(btsc_dir, 'Training'))
        
        random_indices_train = np.random.RandomState(seed=8).permutation(X_train.shape[0])
        X_train = X_train[random_indices_train]; Y_train = Y_train[random_indices_train]
        
        random_indices_val = np.random.RandomState(seed=9).permutation(X_val.shape[0])
        X_val = X_val[random_indices_val]; Y_val = Y_val[random_indices_val]
        
        X_test, Y_test = load_test(os.path.join(btsc_dir, 'Testing'))
        random_indices_test = np.random.RandomState(seed=10).permutation(X_test.shape[0])
        X_test = X_test[random_indices_test]; Y_test = Y_test[random_indices_test]
        
        # normalize data consistently (in case they would not already be)
        all_min, all_max = min_max_data_np([X_train, X_val, X_test])
        X_train = normalize(X_train, all_min, all_max)
        X_val = normalize(X_val, all_min, all_max)
        X_test = normalize(X_test, all_min, all_max)
        
#         make_hist_of_classes(Y_train, "Train")
#         make_hist_of_classes(Y_val, "Validation")
#         make_hist_of_classes(Y_test, "Test")

        return X_train, Y_train, X_val, Y_val, X_test, Y_test


def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)
    return img

def track_no(path):   # returns the track number (as a string of 5 chars) of a given .ppm path 
    return path[-15:-10]

def five_char(n):   # returns a string of 5 char corresponding to n; e.g. 5->'00005', 23->'00023' 
    if(n < 0 or n >= 10**5):
        raise ValueError("The number should be between 0 and 99999")
    elif(n == 0):
        return "00000"
    else:
        no_digits = int(np.log10(n)) + 1
        no_zeros = 5 - no_digits
        return '0' * no_zeros + str(n)


def load_train(data_dir):
    h5filename = os.path.join(data_dir, 'BTSC_Train_and_Validation_correct-split' + str(IMG_SIZE) + '.h5')
    
    try:
        with  h5py.File(h5filename, 'r') as hf:
            X_train, Y_train, X_val, Y_val = hf['train_imgs'][:], hf['train_labels'][:], hf['val_imgs'][:], hf['val_labels'][:]
        
        print("Loaded images from {:}".format(h5filename))

    except (IOError, OSError, KeyError):
        print("Error in reading {:}. Processing all images...".format(h5filename))
        img_root_dir = data_dir
        train_imgs = []
        train_labels = []
        val_imgs = []      
        val_labels = []
        
#         track_val = 0
#         track_total = 0
#         for i in range(NUM_CLASSES):        
#             class_img_paths = glob.glob(os.path.join(img_root_dir, five_char(i) + '/*.ppm'))
            
#             tracknr = []
#             for img in class_img_paths:
#                 tracknr.append(track_no(img))
#             print("Class " + str(i) + " has " + str(len(class_img_paths)) + " images and " + str(len(set(tracknr))) + " tracks")
            
#             track_total += len(set(tracknr))
#             track_val += max(1, int(len(set(tracknr))*0.2))
#         print("No validation tracks: ", track_val, " Total no: ", track_total)
#         import pdb;pdb.set_trace()

        np.random.seed(42)
        for cl in range(NUM_CLASSES): 
            current_dir  = img_root_dir + "/" + five_char(cl) + "/"
            class_img_paths = sorted(glob.glob(os.path.join(current_dir, '*.ppm'))) 
            
            class_tracks = []
            for img_path in class_img_paths:
                class_tracks.append(int(track_no(img_path)))
            
            unique_tracks = np.unique(class_tracks)
            no_tracks_val = max(1, int(len(unique_tracks)*0.2))
            tracks_val = np.random.choice(unique_tracks, no_tracks_val, replace = False)
            
            print("Class ", str(cl), ', No tracks: ', len(unique_tracks), ', Val tracks: ', tracks_val)
            
            np.random.shuffle(class_img_paths)
            
            class_val_paths = [p for p in class_img_paths if int(track_no(p)) in tracks_val]
            class_train_paths = [p for p in class_img_paths if int(track_no(p)) not in tracks_val]
                             
            for img_path in class_train_paths:
                try:
                    img = preprocess_img(io.imread(img_path))
                    img = skimage.transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')
                    train_imgs.append(img)
                    train_labels.append(cl)                     
                except (IOError, OSError):
                    print('missed', img_path)
                    pass
                                 
            for img_path in class_val_paths:
                try:
                    img = preprocess_img(io.imread(img_path))
                    img = skimage.transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')
                    val_imgs.append(img)
                    val_labels.append(cl)                     
                except (IOError, OSError):
                    print('missed', img_path)
                    pass   
        
        X_train = np.array(train_imgs, dtype='float32')        
        Y_train = np.array(train_labels, dtype='int32')
        X_val = np.array(val_imgs, dtype='float32')
        Y_val = np.array(val_labels, dtype='int32')
        
        with h5py.File(h5filename, 'w') as hf:
            hf.create_dataset('train_imgs', data=X_train)
            hf.create_dataset('train_labels', data=Y_train)
            hf.create_dataset('val_imgs', data=X_val)
            hf.create_dataset('val_labels', data=Y_val)

    return X_train, Y_train, X_val, Y_val



def load_test(data_dir):  #normally data_dir='/ssd_data/datasets/BTSC + Training/Testing'
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """

    # Get all subdirectories of data_dir. Each represents a label.

    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            img_np = io.imread(f)
            images.append(preprocess_img(io.imread(f)))
            labels.append(int(d))

    images_newsize = [skimage.transform.resize(image, (IMG_SIZE, IMG_SIZE), mode='constant')
                for image in images]
    images_newsize = np.asarray(images_newsize, dtype = np.float32)
    labels = np.asarray(labels, dtype = np.int32)

    return images_newsize, labels


#     #TODO check me (Riccardo: do not delete, first check!)
#     import ipdb;ipdb.set_trace()
#     import ipdb;ipdb.set_trace()
#     shuffled_images32, shuffled_labels = zip(*np.random.shuffle(list(zip(images32, labels))))

#     return shuffled_images32, shuffled_labels




def make_hist_of_classes(y, text):
    plt.hist(y, bins = np.arange(NUM_CLASSES) + 1, density = True, color = 'm')
    plt.title("Distribution of Classes on " + text)
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.savefig("/data1/temp/" + text + str(len(y)) + str(np.random.randint(0, 100)) + ".png")
    plt.close()

# def load_test(btsc_dir='/ssd_data/datasets/BTSC'):
#     h5filename = os.path.join(btsc_dir, 'BTSC_Test.h5')

#     try:
#         with  h5py.File(h5filename, 'r') as hf:
#             X, Y = hf['imgs'][:], hf['labels'][:]
#         print("Loaded images from {:}".format(h5filename))

#     except (IOError, OSError, KeyError):
#         print("Error in reading {:}. Processing all images...".format(h5filename))

#         img_root_dir = os.path.join(btsc_dir, 'Final_Test/Images/')
#         csvfilename = os.path.join(img_root_dir, 'GT-final_test.csv')

#         test = pd.read_csv(csvfilename, sep=';')

#         # Load test dataset
#         X = []
#         Y = []
#         for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
#             img_path = os.path.join(img_root_dir, file_name)
#             X.append(preprocess_img(io.imread(img_path)))
#             Y.append(class_id)

#         X = np.array(X, dtype='float32')
#         Y = np.array(Y, dtype='int32')

#         with h5py.File(h5filename, 'w') as hf:
#             hf.create_dataset('imgs', data=X)
#             hf.create_dataset('labels', data=Y)

#     return X, Y
