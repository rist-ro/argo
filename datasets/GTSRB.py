"""
Module for managing GTSRB dataset
"""

import numpy as np

import os
import urllib


from .ImageDataset import ImageDataset

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

import matplotlib
import matplotlib.pyplot as plt

NUM_CLASSES = 43
IMG_SIZE = 32

# IF WANT TO USE A DIFFERENT IMAGE SIZE OR WANT TO MAKE NEW TRAIN AND TEST SETS, YOU SOHULD FIRST RUN IT ON IRONMAN, AS IT'S THE ONLY WORKSTATION
# THAT HAS THE RAW IMAGES SAVED (/ssd_data/datasets/GTSRG/Final_Training/Images etc.)


class GTSRB(ImageDataset):
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

        default_data_dir = '/ssd_data/datasets/GTSRB'

        self.data_dir = self._params['data_dir'] if 'data_dir' in params else default_data_dir
                
        self._train_set_x, self._train_set_y, \
            self._validation_set_x,  self._validation_set_y, \
            self._test_set_x,  self._test_set_y = self.load_data(self.data_dir)


    @staticmethod
    def dataset_id(params):
        """
        This method interprets the parameters and generate an id
        """

        GTSRB.check_params_impl(params)

        id = 'GTSRB'

        return id

#     @staticmethod
#     def load_data(gtsrb_dir):

#         try:
#             os.stat(gtsrb_dir)
#         except:
#             os.mkdir(gtsrb_dir)

#         # import pdb;pdb.set_trace()
#         X_shuffled, Y_shuffled = load_train(gtsrb_dir)
#         val_frac = 0.15
#         X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_shuffled, Y_shuffled,
#                                                                                   test_size=val_frac, shuffle=False, random_state = 42)
#         X_test, Y_test = load_test(gtsrb_dir)

#         # normalize data consistently (in case they would not already be)
#         all_min, all_max = min_max_data_np([X_train, X_val, X_test])
#         X_train = normalize(X_train, all_min, all_max)
#         X_val = normalize(X_val, all_min, all_max)
#         X_test = normalize(X_test, all_min, all_max)
        
# #         make_hist_of_classes(Y_train, "Train")
# #         make_hist_of_classes(Y_val, "Validation")
# #         make_hist_of_classes(Y_test, "Test")

#         return X_train, Y_train, X_val, Y_val, X_test, Y_test


    @staticmethod
    def load_data(gtsrb_dir):

        try:
            os.stat(gtsrb_dir)
        except:
            os.mkdir(gtsrb_dir)

        X_train, Y_train, X_val, Y_val = load_train(gtsrb_dir)
        
        random_indices_train = np.random.RandomState(seed=8).permutation(X_train.shape[0])
        X_train = X_train[random_indices_train]; Y_train = Y_train[random_indices_train]
        
        random_indices_val = np.random.RandomState(seed=9).permutation(X_val.shape[0])
        X_val = X_val[random_indices_val]; Y_val = Y_val[random_indices_val]
        
        X_test, Y_test = load_test(gtsrb_dir)
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



# from https://chsasank.github.io/keras-tutorial.html
def get_class(img_path):
    return int(img_path.split('/')[-2])

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

# #         import pdb;pdb.set_trace()
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

#     return X, Y



def load_train(gtsrb_dir='/ssd_data/datasets/GTSRB'):
    h5filename = os.path.join(gtsrb_dir, 'GTSRB_Train_and_Validation_correct-split' + str(IMG_SIZE) + '.h5')
    
    try:
        with  h5py.File(h5filename, 'r') as hf:
            X_train, Y_train, X_val, Y_val = hf['train_imgs'][:], hf['train_labels'][:], hf['val_imgs'][:], hf['val_labels'][:]
        print("Loaded images from {:}".format(h5filename))

    except (IOError, OSError, KeyError):
        print("Error in reading {:}. Processing all images...".format(h5filename))
        img_root_dir = os.path.join(gtsrb_dir, 'Final_Training/Images/')
        train_imgs = []
        train_labels = []
        val_imgs = []      
        val_labels = []
        
#         for i in range(43):        
#             print(len(glob.glob(os.path.join(img_root_dir, five_char(i) + '/*.ppm')))%30)
#         print("--------------------")
#         for j in range(23):
#             print(len(glob.glob(os.path.join(img_root_dir, '00033/' + five_char(j) + '*.ppm')))) track 19 has 29 imgs

        np.random.seed(42)
        for cl in range(NUM_CLASSES):
            current_dir  = img_root_dir + five_char(cl) + "/"
            class_img_paths = sorted(glob.glob(os.path.join(current_dir, '*.ppm'))) 
            
            max_tr = int(track_no(class_img_paths[-1]))  #get the track number of the last image
            val_tr = np.random.randint(0, max_tr)
            print("Class " + str(cl) + ', Val track ' + str(val_tr) + ', Max track ' + str(max_tr))
            
            np.random.shuffle(class_img_paths)
            
            class_val_paths = [p for p in class_img_paths if track_no(p) == five_char(val_tr)]
            class_train_paths = [p for p in class_img_paths if track_no(p) != five_char(val_tr)]
                             
            for img_path in class_train_paths:
                try:
                    img = preprocess_img(io.imread(img_path))
                    train_imgs.append(img)
                    train_labels.append(cl)                     
                except (IOError, OSError):
                    print('missed', img_path)
                    pass
                                 
            for img_path in class_val_paths:
                try:
                    img = preprocess_img(io.imread(img_path))
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



def load_test(gtsrb_dir='/ssd_data/datasets/GTSRB'):
    h5filename = os.path.join(gtsrb_dir, 'GTSRB_Test' + str(IMG_SIZE) + '.h5')

    try:
        with  h5py.File(h5filename, 'r') as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]
        print("Loaded images from {:}".format(h5filename))

    except (IOError, OSError, KeyError):
        print("Error in reading {:}. Processing all images...".format(h5filename))

        img_root_dir = os.path.join(gtsrb_dir, 'Final_Test/Images/')
        csvfilename = os.path.join(img_root_dir, 'GT-final_test.csv')

        test = pd.read_csv(csvfilename, sep=';')

        # Load test dataset
        X = []
        Y = []
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join(img_root_dir, file_name)
            X.append(preprocess_img(io.imread(img_path)))
            Y.append(class_id)

        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='int32')

        with h5py.File(h5filename, 'w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y)

    return X, Y


       
        
def make_hist_of_classes(y, text):
    plt.hist(y, bins = np.arange(NUM_CLASSES) + 1, density = True, color = 'm')
    plt.title("Distribution of Classes on " + text)
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.savefig("/data1/temp/" + text + str(len(y)) + str(np.random.randint(0, 100)) + ".png")
    plt.close()
