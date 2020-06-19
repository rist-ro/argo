import numpy as np
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf

from core.VAELauncher import VAELauncher
from core import VAE

from argo.core.TFModelSaver import TFModelSaver
from argo.core.ArgoLauncher import make_list
#from vae import input_data

from datasets.Dataset import Dataset
from datasets.MNIST import MNIST
from argo.core.ImagesGenerator import ImagesGenerator

from itertools import product

from scipy.special import expit

import sys
import argparse

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth=True

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vae_conf', help = 'Conf file of VAE model')
parser.add_argument('perturbed_examples', help = 'File of perturbed examples')
parser.add_argument('label_perturbed_examples', help = 'Word to describe the perturbed examples, e.g. adversarial FF, noisy etc.; will appear on histograms')
parser.add_argument('--clean_examples', '-cl', help = 'File of clean examples', default = '../petru_foca/files/adversarial_examples/tensors_rank_2/clean.npy')


args = parser.parse_args()

vae_conf = args.vae_conf
perturbed_examples = args.perturbed_examples
clean_examples = args.clean_examples
label_perturbed_examples = args.label_perturbed_examples


X_clean = np.load(clean_examples)
X_pert = np.load(perturbed_examples)
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", X_pert.shape)

# ---------------------------- LOADING VAE -----------------------------------------------------

launcher = VAELauncher()
parallelism = 1 # 1 is equivalent to pool
dataset_conf, params, config =  launcher.process_conf_file(vae_conf)
dataset = Dataset.load_dataset(dataset_conf)
# train
launcher.run(dataset, params, config, parallelism) # uncomment this line
# model is now trained
print("Model(s) trained")
# pdb.set_trace()

# create the object, it is empty
run = 0 # run of the algorithm
epoch = 1000 # checkpoint from which to load
gpu = 0 # number of the GPU to which I want to allocate the tensorflow graph (-1 = CPU)

# load the model
vae = TFModelSaver.restore_model_from_conf_filename(VAE.GaussianVariationalAutoEncoder, vae_conf, epoch, gpu=gpu)

#------------------------------------------------------------------------------------------------------

mu_clean = vae.encode(X_clean)[0]
mu_pert = vae.encode(X_pert)[0]


"""
# Plot histogram of mu(clean) and mu(adv)
for i in range(20):
    clean = mu_clean[:,i]
    adv = mu_adv[:,i]

    min_clean = np.amin(clean); max_clean = np.amax(clean)
    min_adv = np.amin(adv); max_adv = np.amax(adv)


    fig = plt.figure(figsize = (10, 15))
    h1 = fig.add_subplot(121)
    h2 = fig.add_subplot(122)


    bins1 = np.linspace(-4 ,4, 101)
    h1.hist(clean, bins1, color = 'b', label='Clean, component ' + str(i))
    h1.legend(loc='upper right')

    bins2 = np.linspace(-4 ,4, 101)
    h2.hist(adv, bins2, color = 'b', label='Adversarial, component ' + str(i))
    h2.legend(loc='upper right')
    #plt.show()

    #plt.savefig("../files/jan2018_advex/vae_ff_sampling_advex/cw_vaeff_hist_mean_component_" + str(i) + ".png")
    #plt.savefig("../files/jan2018_advex/jun1/clean_reconstr/mean_clean_reconstr_99_" + str(i) + ".png")



# Plot histogram of mu(clean) - mu(adv)
for i in range(20):
    clean = mu_clean[:,i]
    adv = mu_adv[:,i]

    diff = (mu_clean - mu_adv)[:,i]
    min_diff = np.amin(diff); max_diff = np.amax(diff)

    fig2 = plt.figure(figsize = (7, 7))
    h = fig2.add_subplot(111)

    bins = np.linspace(min_diff, max_diff, 101)
    h.hist(diff, bins, color = 'r', label = 'Difference, component ' + str(i))
    h.legend(loc='upper right')
    #plt.savefig("../files/jan2018_advex/vae_ff_sampling_advex/diff_hist_mean_component_" + str(i) + ".png")
    #plt.savefig("../files/jan2018_advex/jun1/clean_reconstr/diff_clean_reconstr_99_" + str(i) + ".png")
    #plt.show()
"""

# Lipschitz Constant

norm_dist_clean_pert = np.linalg.norm(mu_clean - mu_pert, axis = 1)
image_dist_clean_pert = np.linalg.norm(X_clean - X_pert, axis = 1)

lipschitz_clean_pert = [norm_dist_clean_pert[i]/image_dist_clean_pert[i] for i in range(10000) if np.abs(image_dist_clean_pert[i]) > 10**(-9)]

print(len(lipschitz_clean_pert))

bins_lipschitz = np.linspace(0.6,3.6,100)

plt.hist(lipschitz_clean_pert, bins_lipschitz, color = 'g', label = "Clean - " + label_perturbed_examples)
plt.title("Lipschitz constant")
plt.legend(loc='upper right')
plt.savefig("./lipschitz_" + label_perturbed_examples + ".png")
plt.show()
