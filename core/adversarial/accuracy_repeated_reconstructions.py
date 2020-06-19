import numpy as np
import tensorflow as tf
import pdb
import argparse
from tensorflow.examples.tutorials.mnist import input_data


from datasets.Dataset import Dataset
from core.argo.core.ArgoLauncher import ArgoLauncher, get_full_id
from core.argo.core.TrainingLauncher import TrainingLauncher
from core.argo.core.TFDeepLearningModel import load_model, load_network
from vae.core.VAE import GaussianVariationalAutoEncoder
from prediction.core.PredictionModel import get_prediction_model

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='2'
# sess_config = tf.ConfigProto()
# sess_config.gpu_options.allow_growth=True

parser = argparse.ArgumentParser(description='Compute accuracy of repeated reconstructions through VAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model_choice', help = 'The model to use as classifier; choose between: ff, vaeffnos, vaeffs, vffnos, vffs')
parser.add_argument('ffconffile', help = 'The config file associated to the training of the Prediction model to load.')
parser.add_argument('folder_to_load', help = 'The folder where the reconstructions are saved')
parser.add_argument('--data_rank', '-rk', help = 'The tensor rank of data; rk = 2 for MLP, rk = 4 for Conv.', default = 4)
parser.add_argument('--autoencconffile', '-ae', help = 'The config file associated to the training of the autoencoder model to load.', default = None)
parser.add_argument('--global_step_vae', help = 'The global_step at which we want to restore VAE model. Default is the last one found in the folder.', default=None)
parser.add_argument('--global_step_ff', help = 'The global_step at which we want to restore Prediction model. Default is the last one found in the folder.', default=None)
parser.add_argument('--gpu', '-gpu', help = 'GPU where to run on.', default = '0')

args = parser.parse_args()
model_choice = args.model_choice
ffconffile = args.ffconffile
folder_to_load = args.folder_to_load
data_rank = int(args.data_rank)
autoencconffile = args.autoencconffile
global_step_vae = args.global_step_vae
global_step_ff = args.global_step_ff
gpu = args.gpu

seed = 120
parallelism = 0 # 0 is equivalent to single



###################
### SOME CHECKS ###
###################

# FF has: logits
# VAEFF has: mean, cov, reconstr, logits
# VFF has: mean, cov, logits

if(model_choice not in ['ff', 'vaeffnos', 'vaeffs', 'vffnos', 'vffs']):
    raise ValueError("Choose a correct model!")

if(model_choice[0] == 'v'):
    if(autoencconffile is None):
        raise ValueError("Either you chose the wrong model or you didn't specify an autoencoder conf file!")

if(model_choice[0] == 'f'):
    if(autoencconffile is not None):
        raise ValueError("FF requires no autoencoder!")



###################
## LOAD DATASETS ##
###################

ffmodeldir = os.path.dirname(ffconffile)
ff_dataset_conf, ff_model_parameters, ff_config = ArgoLauncher.process_conf_file(args.ffconffile)
ff_dataset = Dataset.load_dataset(ff_dataset_conf)

if(model_choice[0] == 'v'): #i.e. there is a VAE model
    vaemodeldir = os.path.dirname(autoencconffile)
    vae_dataset_conf, vae_model_parameters, vae_config = ArgoLauncher.process_conf_file(args.autoencconffile)
    vae_dataset = Dataset.load_dataset(vae_dataset_conf)

    #if datasets x_shape are different raise Exception! what is the meaning of a comparison otherwise?
    assert ff_dataset.x_shape == vae_dataset.x_shape, \
            "the VAE and FF network that you are trying to load have been \
            trained on datasets with different x_shape : `%s` and `%s`"%(str(ff_dataset.x_shape), str(vae_dataset.x_shape))


x_shape = (None,) + ff_dataset.x_shape

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
X_train = np.copy(mnist.train.images) #copy the dataset because it will be shuffled afterwards
y_train = np.copy(mnist.train.labels)
X_test = np.copy(mnist.test.images)
y_test = np.copy(mnist.test.labels)


# Load the 10 datasets
X_np = np.zeros((10, 10000, 784), dtype = 'float32')
for i in range(10):
    X_np[i] = np.load(folder_to_load + "no" + str(i+1) + ".npy")

if(data_rank == 4):
    X_np = X_np.reshape((10, 10000, 28, 28, 1))



################################
## VAE MODEL, GET VAE NETWORK ##
################################

#TODO maybe train here if it has not been trained... ?
# launcher = TrainingLauncher(ModelClass, dataset)
# launcher.run(model_parameters, config, parallelism)

if(model_choice[0] == 'v'): #i.e. there is a VAE model
    vae_run = vae_model_parameters["run"]
    vae_seed = vae_model_parameters["seed"]

    vae_network, vae_checkpoint_name = load_network(GaussianVariationalAutoEncoder, autoencconffile, global_step_vae)



######################################
## PREDICTION MODEL, GET FF NETWORK ##
######################################

#TODO as above, maybe make training.. check if works.. ?
# launcher = TrainingLauncher(ModelClass, dataset)
# launcher.run(model_parameters, config, parallelism)

ff_run = ff_model_parameters["run"]
ff_seed = ff_model_parameters["seed"]
ff_task = ff_model_parameters["task"]

ff_network, ff_checkpoint_name = load_network(get_prediction_model, ffconffile, global_step_ff)



##############################
## SESSION AND CALCULATIONS ##
##############################

# SET THE SEED
tf.set_random_seed(seed)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config = sess_config)

x = tf.placeholder(tf.float32, shape=x_shape, name='input')

# LOAD FF NETWORK
logits = ff_network(x)
ff_network.restore(sess, ff_checkpoint_name)


# LOAD VAE NETWORK
if(model_choice[0] == 'v'): #i.e. there is a VAE model
    # LOAD VAE NETWORK
    model_latent, _, model_visible = vae_network(x)
    vae_network.restore(sess, vae_checkpoint_name)

    n_z_samples = vae_network.n_z_samples

    # CALLABLES FOR CLEVERHANS
    encoder_module = vae_network.encoder_module
    decoder_module = vae_network.decoder_module


# CALLABLES FOR CLEVERHANS
ff_module = ff_network.module



##############################
###### DEFINE THE MODELS #####
##############################

def ff(x):
    logits = ff_module(x)
    return logits, None

def vae_ff_nos(x):
    model_latent = encoder_module(x)
    mean = model_latent.mean()
    rec_distr = decoder_module(mean)
    recnode = rec_distr.reconstruction_node()
    logits = ff_module(recnode)
    return logits, mean, _, recnode # "_" was originally "cov"

def vae_ff_s(x):
    model_latent = encoder_module(x)
    mean = model_latent.mean()
    std = model_latent.scale
    cov = tf.square(std)
    distr = tf.distributions.Normal(loc = mean, scale = std)
    samples = distr.sample(1); samples = tf.reshape(samples, (10000, 20))
    rec_distr = decoder_module(samples)
    recnode = rec_distr.reconstruction_node()
    logits = ff_module(recnode)
    return logits, mean, cov, recnode

def v_ff_nos(x):
    model_latent = encoder_module(x)
    mean = model_latent.mean()
    cov = model_latent.covariance()
    logits = ff_module(mean)
    return logits, mean, cov

def v_ff_s(x):
    model_latent = encoder_module(x)
    mean = model_latent.mean()
    cov = model_latent.covariance()
    distr = tf.distributions.Normal(loc = mean, scale = cov)
    samples = distr.sample(1)
    logits = ff_module(samples)
    return logits, mean, cov




# Set the model function according to our choice
if(model_choice == 'ff'):
    model_function = ff
elif(model_choice == 'vaeffnos'):
    model_function = vae_ff_nos
elif(model_choice == 'vaeffs'):
    model_function = vae_ff_s
elif(model_choice == 'vffnos'):
    model_function = v_ff_nos
elif(model_choice == 'vffs'):
    model_function = v_ff_s


# Compute the 10 accuracy numbers
y_tf = tf.constant(y_test)
accuracy_np = [0. for i in range(10)]

for i in range(10):
    X_tf = tf.constant(X_np[i])
    pdb.set_trace()
    logits_X_tf = model_function(X_tf)[0]

    accuracy = tf.reduce_mean(tf.cast(
               tf.equal(tf.argmax(logits_X_tf, axis = 1),
                     tf.cast(tf.argmax(y_tf, axis = 1), dtype = tf.int64)),
                     dtype = tf.float32))
    accuracy_np[i] = accuracy.eval(session = sess)

print("Accuracy array:", accuracy_np)
