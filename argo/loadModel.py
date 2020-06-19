import sys, os
sys.path.insert(0, os.getcwd())
from argo.core.ArgoLauncher import ArgoLauncher
from argo.core.TFDeepLearningModel import load_model, load_network
from argo.core.utils.argo_utils import load_class
from datasets.Dataset import Dataset
import argparse
import tensorflow as tf
from pprint import pprint
import os

parser = argparse.ArgumentParser(description='Load a Argo Model, first ensures it has been trained.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conffile', help='The config file associated to the training of the model to load.')
parser.add_argument('--global_step', help='The global_step at which we want to restore the model. Default is the last one found in the folder.', default=None)
parser.add_argument('--gpu', help='GPU where to run on.', default='0')
parser.add_argument('--seed', type=int, help='seed to randomize tables.', default=0)
parser.add_argument('--modelClassDir', '-md', help='directory where where we can find the file containing the class of the model to load (relative to where I launch this script, e.g. `core` or `prediction.core`, ...)', default="core")

args = parser.parse_args()

gpu = args.gpu
seed = args.seed
global_step = args.global_step

conffile = args.conffile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

model_class_base_path = args.modelClassDir
model_class_base_path.replace("/", ".")

# ######################################################
# # LOAD THE WHOLE MODEL WITH ITS OWN MONITOREDSESSION #
# ######################################################
#

model, dataset = load_model(conffile,
                            global_step=global_step,
                            gpu=gpu,
                            seed=seed,
                            model_class_base_path=model_class_base_path)

network = model._network

all_variables = network.get_all_variables()
pprint(all_variables)
checkpoint_name = model.checkpoint_name(global_step)
all_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]

# pdb is here to play. Enjoy! (Riccardo)
import pdb;pdb.set_trace()

#clean
model.sess.close()
tf.reset_default_graph()


##################################################
# LOAD THE NETWORK AND RESTORE IN MY OWN SESSION #
##################################################

dataset_conf, model_parameters, config = ArgoLauncher.process_conf_file(conffile)
dataset = Dataset.load_dataset(dataset_conf)
x_shape = (None,)+dataset.x_shape

ModelClass = load_class(model_parameters["model"], base_path=model_class_base_path)

network, checkpoint_name = load_network(ModelClass, conffile, global_step=global_step, dataset=dataset)

# SET THE SEED
tf.set_random_seed(seed)
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config = sess_config)

x = tf.placeholder(tf.float32, shape=x_shape, name='input')

# LOAD NETWORK
stuffs = network(x)
all_variables = network.get_all_variables()
network.restore(sess, checkpoint_name)
pprint(all_variables)
