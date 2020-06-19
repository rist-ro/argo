"""
Module for managing GTSRB_SS dataset download from https://www.towardsautonomy.com/perception/traffic_sign_classification
"""

import numpy as np

import os
import urllib


from .ImageDataset import ImageDataset
import pickle
import h5py
import sklearn
import sklearn.model_selection
from skimage import io
import os
import pandas as pd
from skimage import color, exposure, transform
import glob
import numpy as np
from .utils import normalize, min_max_data_np

NUM_CLASSES = 43
IMG_SIZE = 32

# IF WANT TO USE A DIFFERENT IMAGE SIZE OR WANT TO MAKE NEW TRAIN AND TEST SETS, YOU SOHULD FIRST RUN IT ON IRONMAN, AS IT'S THE ONLY WORKSTATION
# THAT HAS THE RAW IMAGES SAVED (/ssd_data/datasets/GTSRG/Final_Training/Images etc.)


class GTSRB_SS(ImageDataset):
    """
    This class manage the dataset GTSRB, properties of the datasets are uniquely determined
    by the params dictionary

    """

    default_params = {
        # "something" : param
        }

    def __init__(self, params):
        super().__init__(params)

        self._id = self.dataset_id(params)

        default_data_dir = '/ssd_data/Traffic_signs_test/'

        self.data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir

        self._train_set_x, self._train_set_y, \
            self._validation_set_x,  self._validation_set_y, \
            self._test_set_x,  self._test_set_y = self.load_data(self.data_dir)


    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        Traffic_signs_test.check_params_impl(params)

        id = 'GTSRB_SS'

        return id

    @staticmethod
    def load_data(gtsrb_dir):

        try:
            os.stat(gtsrb_dir)
        except:
            os.mkdir(gtsrb_dir)

        # import pdb;pdb.set_trace()
        #X_shuffled, Y_shuffled = load_train(gtsrb_dir)
        #val_frac = 0.15
        #X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_shuffled, Y_shuffled,
        #                                                                          test_size=val_frac, shuffle=False)
        X_train, Y_train=import_data(gtsrb_dir, "train")
        X_test,  Y_test=import_data(gtsrb_dir, "test")
        X_val,   Y_val=import_data(gtsrb_dir, "valid")
        
        
        
        
        #X_test, Y_test = load_test(gtsrb_dir)

        # normalize data consistently (in case they would not already be)
        all_min, all_max = min_max_data_np([X_train, X_val, X_test])
        X_train = normalize(X_train, all_min, all_max)
        X_val = normalize(X_val, all_min, all_max)
        X_test = normalize(X_test, all_min, all_max)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test


# from https://chsasank.github.io/keras-tutorial.html
def get_class(img_path):
    return int(img_path.split('/')[-2])


def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    return img



def import_data(gtsrb_dir='/ssd_data/', data="train"):
    data_file = os.path.join(gtsrb_dir, '{}'.format(data)+'.p' )
    with open(data_file, mode='rb') as f:
        data_file_numpy  = pickle.load(f)
    X, Y = np.float32(data_file_numpy['features']), data_file_numpy['labels']
    return X, Y
                              
                              


# def load_train(gtsrb_dir='/ssd_data/datasets/GTSRB'):
#     h5filename = os.path.join(gtsrb_dir, 'GTSRB_Train_and_Validation_shuffled' + str(IMG_SIZE) + '.h5')

#     try:
#         with  h5py.File(h5filename, 'r') as hf:
#             X, Y = hf['imgs'][:], hf['labels'][:]
#         print("Loaded images from {:}".format(h5filename))

#     except (IOError, OSError, KeyError):
#         print("Error in reading {:}. Processing all images...".format(h5filename))
#         img_root_dir = os.path.join(gtsrb_dir, 'Final_Training/Images/')
#         imgs = []
#         labels = []

#         all_img_paths = glob.glob(os.path.join(img_root_dir, '*/*.ppm'))
#         np.random.shuffle(all_img_paths)
#         for img_path in all_img_paths:
#             try:

#                 img = preprocess_img(io.imread(img_path))
#                 label = get_class(img_path)
#                 imgs.append(img)
#                 labels.append(label)

#                 if len(imgs) % 1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
#             except (IOError, OSError):
#                 print('missed', img_path)
#                 pass

#         X = np.array(imgs, dtype='float32')
#         Y = np.array(labels, dtype='int32')

#         with h5py.File(h5filename, 'w') as hf:
#             hf.create_dataset('imgs', data=X)
#             hf.create_dataset('labels', data=Y)

#     return X, Y


# def load_test(gtsrb_dir='/ssd_data/datasets/GTSRB'):
#     h5filename = os.path.join(gtsrb_dir, 'GTSRB_Test' + str(IMG_SIZE) + '.h5')

#     try:
#         with  h5py.File(h5filename, 'r') as hf:
#             X, Y = hf['imgs'][:], hf['labels'][:]
#         print("Loaded images from {:}".format(h5filename))

#     except (IOError, OSError, KeyError):
#         print("Error in reading {:}. Processing all images...".format(h5filename))

#         img_root_dir = os.path.join(gtsrb_dir, 'Final_Test/Images/')
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
