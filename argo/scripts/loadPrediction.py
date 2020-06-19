raise Exception("I am deprecated, look at `argo/loadModel.py` instead!")

from core.Prediction import get_prediction_model

parser = argparse.ArgumentParser(description='Load a Prediction model, first ensures it has been trained.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conffile', help='The config file associated to the training of the model to load.')
parser.add_argument('--global_step', help='The global_step at which we want to restore the model. Default is the last one found in the folder.', default=None)
parser.add_argument('--gpu', help='GPU where to run on.', default='0')
parser.add_argument('--seed', type=int, help='seed to randomize tables.', default=0)
parser.add_argument('--savenet', action='store_true', help='wheter to save the network')
parser.add_argument('--dirSaveFFNetwork', '-d', help='directory where to save FF Network. If not specified the network will be saved in the model directory', default=None)

args = parser.parse_args()

gpu = args.gpu
seed = args.seed
global_step = args.global_step

network_dir = args.dirSaveFFNetwork
save_net = args.savenet
conffile = args.conffile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


# ######################################################
# # LOAD THE WHOLE MODEL WITH ITS OWN MONITOREDSESSION #
# ######################################################
#
pred_model, dataset = load_model(get_prediction_model, conffile,
                        global_step=global_step, save_net=save_net, network_dir=network_dir, gpu=gpu, seed=seed)
#
# # ff_network = pred_model._network
# # checkpoint_name = pred_model.checkpoint_name(global_step)
#

import pdb; pdb.set_trace()

#clean
pred_model.sess.close()
tf.reset_default_graph()

##################################################
# LOAD THE NETWORK AND RESTORE IN MY OWN SESSION #
##################################################

dataset_conf, model_parameters, config = ArgoLauncher.process_conf_file(conffile)

dataset = Dataset.load_dataset(dataset_conf)

ff_network, ff_checkpoint_name = load_network(get_prediction_model, conffile, global_step=global_step, dataset=dataset)

x_shape = (None,)+dataset.x_shape

# SET THE SEED
tf.set_random_seed(seed)
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config = sess_config)

x = tf.placeholder(tf.float32, shape=x_shape, name='input')

# LOAD FF NETWORK
logits = ff_network(x)
# ff_variables = ff_network.get_all_variables()
ff_all_variables = ff_network.get_all_variables()
ff_network.restore(sess, ff_checkpoint_name)

pprint(ff_all_variables)
