from core.VAELauncher import VAELauncher
from core import VAE

from argo.core.TFModelSaver import TFModelSaver
from argo.core.ArgoLauncher import make_list
#from vae import input_data

from datasets.Dataset import Dataset
from datasets.MNIST import MNIST
from argo.core.ImagesGenerator import ImagesGenerator

import numpy as np

#matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pdb

from itertools import product

import tensorflow as tf

from scipy.special import expit

import sys

#from testingVAE_tools.l2_average_error import l2_reconstr_error
from testingVAE_tools.accuracy_testing import ff, accuracy



#folder to save weights
vae_dir = sys.argv[1]


# ---------------------------- LOADING VAE -----------------------------------------------------

launcher = VAELauncher()
parallelism = 1 # 1 is equivalent to pool
dataset_conf, params, config =  launcher.process_conf_file(vae_dir)
dataset = Dataset.load_dataset(dataset_conf)
# train
launcher.run(dataset, params, config, parallelism) # uncomment this line
# model is now trained
print("Model(s) trained")


# create the object, it is empty
run = 0 # run of the algorithm
epoch = 1000 # checkpoint from which to load
gpu = 0 # number of the GPU to which I want to allocate the tensorflow graph (-1 = CPU)

# load the model
vae = TFModelSaver.restore_model_from_conf_filename(VAE.GaussianVariationalAutoEncoder, vae_dir, epoch, gpu=gpu)

#------------------------------------ END OF LOADING VAE ---------------------------------------




dir_adv = "../petru_foca/files/adversarial_examples/"
clean_test = np.load(dir_adv + "clean_test.npy").reshape((10000,784))

def l2_reconstr_error(X, no_passes):
    errors = np.zeros((no_passes, np.shape(X)[0]))
    for i in range(no_passes):
        reconstr,_,_,_,_ = vae.reconstruct(X)
        l2_error = np.linalg.norm(X - reconstr, ord = 2, axis = 1)
        errors[i] = l2_error
    return np.average(errors)


no_passes = 10
l2_average_error = l2_reconstr_error(clean_test, no_passes)
print("The average L2 error between images and reconstructions:", l2_average_error)


#  ------------------- ACCURACY OF RECONSTRUCTIONS ON FF ----------------------
network_dir = "../petru_foca/files/my_data/"

W0_np = np.load(network_dir + 'W0.npy')
W1_np = np.load(network_dir + 'W1.npy')
b0_np = np.load(network_dir + 'b0.npy')
b1_np = np.load(network_dir + 'b1.npy')


weights = [W0_np, W1_np]
biases = [b0_np, b1_np]


#dir_adv = "../petru_foca/files/adversarial_examples/"
#clean_test = np.load(dir_adv + "clean_test.npy").reshape((10000,784))
cw_test = np.load(dir_adv + "cw_mydata_test.npy").reshape((10000,784))
fgsm01_test = np.load(dir_adv + "fgsm_mydata_eps01_test.npy").reshape((10000,784))
fgsm03_test = np.load(dir_adv + "fgsm_mydata_eps03_test.npy").reshape((10000,784))

reconstr_clean_test = vae.reconstruct(clean_test)[0]
reconstr_cw_test = vae.reconstruct(clean_test)[0]
reconstr_fgsm01_test = vae.reconstruct(clean_test)[0]
reconstr_fgsm03_test = vae.reconstruct(clean_test)[0]

acc_clean_test = accuracy(reconstr_clean_test, weights, biases)
acc_cw_test = accuracy(reconstr_cw_test, weights, biases)
acc_fgsm01_test = accuracy(reconstr_fgsm01_test, weights, biases)
acc_fgsm03_test = accuracy(reconstr_fgsm03_test, weights, biases)

print("Accuracy on clean   :", acc_clean_test)
print("Accuracy on CW      :", acc_cw_test)
print("Accuracy on FGSM 0.1:", acc_fgsm01_test)
print("Accuracy on FGSM 0.3:", acc_fgsm03_test)
print("(All adversarial examples were obtained with \"my_data\" network)")


# --------------------- CONTRACTIVENESS --------------------------------------
