raise Exception("I am deprecated, look at `argo/loadModel.py` instead!")

parser = argparse.ArgumentParser(description='Load a VAE model, first ensures it has been trained.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('conffile', help='The config file associated to the training of the model to load.')
parser.add_argument('--global_step', help='The global_step at which we want to restore the model. Default is the last one found in the folder.', default=None)
parser.add_argument('--gpu', help='GPU where to run on.', default='0')
parser.add_argument('--seed', type=int, help='seed to randomize tables.', default=0)
parser.add_argument('--savenet', action='store_true', help='wheter to save the network')
parser.add_argument('--dirSaveVAENetwork', '-d', help='directory where to save VAE Network. If not specified the network will be saved in the model directory', default=None)

args = parser.parse_args()

gpu = args.gpu
seed = args.seed
global_step = args.global_step

network_dir = args.dirSaveVAENetwork
save_net = args.savenet

conffile = args.conffile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

model_class_base_path = "core" #it specifies the folder where we can find the file containing the class of the model to load

# ######################################################
# # LOAD THE WHOLE MODEL WITH ITS OWN MONITOREDSESSION #
# ######################################################
#

vae_model, dataset = load_model(conffile, global_step=global_step,
                                save_net=save_net,
                                network_dir=network_dir,
                                gpu=gpu,
                                seed=seed,
                                model_class_base_path=model_class_base_path)

vae_network = vae_model._network
vae_variables = vae_network.get_all_variables()
pprint(vae_variables)

# checkpoint_name = vae_model.checkpoint_name(global_step)

#clean
vae_model.sess.close()
tf.reset_default_graph()


##################################################
# LOAD THE NETWORK AND RESTORE IN MY OWN SESSION #
##################################################

dataset_conf, model_parameters, config = ArgoLauncher.process_conf_file(conffile)
dataset = Dataset.load_dataset(dataset_conf)
x_shape = (None,)+dataset.x_shape

ModelClass = load_class(model_parameters["model"], base_path=model_class_base_path)

vae_network, vae_checkpoint_name = load_network(ModelClass, conffile, global_step=global_step, dataset=dataset)


# SET THE SEED
tf.set_random_seed(seed)
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config = sess_config)

x = tf.placeholder(tf.float32, shape=x_shape, name='input')

# LOAD NETWORK
stuffs = vae_network(x)
vae_variables = vae_network.get_all_variables()
vae_network.restore(sess, vae_checkpoint_name)
pprint(vae_variables)
