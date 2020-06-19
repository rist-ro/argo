from .argo.core.ArgoLauncher import ArgoLauncher
from .argo.core.Launchable import Launchable

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import os

from .transform.transform import get_transform_module, get_transform_id, check_dataset_shapes
from .attack.attack import get_attack_class, get_attack_id

from datasets.Dataset import Dataset

from .argo.core.TFDeepLearningModel import load_network, update_model_params
from .argo.core.utils.argo_utils import load_class, unpack_dict_of_lists, apply_resize

import pandas as pd
from glob import glob

from matplotlib import pyplot as plt

from .argo.core.utils.ImagesSaver import ImagesSaver

import pdb

from tqdm import tqdm
import time
from functools import partial

import re
from .transform.transform import load_network_with_dummy_x

def load_model(conf_file, dataset=None, gpu=0, seed=0, model_class_base_path=''):
    """Load a TFDeepLearningModel and optionally save its network

    Args:
        conf_file (str): the conf file of the model where to find the experiment.
        dataset (datasets.Dataset): (optional) the argo Dataset of the model for the training. If not passed it will be reloaded.
        global_step (int): the global step to load the checkpoint (if None the last checkpoint present will be loaded).
        gpu (int) : the gpu on which the model will create the session
        seed (int) : the seed that the model will set
        model_class_base_path (str): the base path where to look for the model class

    Returns:
        TFDeepLearningModel: The loaded Argo TFDeepLearningModel.
        datasets.Dataset: the argo Dataset of the model for the training.

    """

    dataset_conf, model_parameters, config = ArgoLauncher.process_conf_file(conf_file)

    if not dataset:
        dataset = Dataset.load_dataset(dataset_conf)

    ArgoTFDeepLearningModelClass = load_class(model_parameters["model"], base_path=model_class_base_path)

    update_model_params(model_parameters, dataset)

    # baseDir = config["dirName"]+"/"+dataset.id
    model_dir = os.path.split(os.path.dirname(conf_file))[0]
    model = ArgoTFDeepLearningModelClass(model_parameters, model_dir, gpu=gpu, seed=seed)

    return model, dataset


class AdversarialModel(Launchable):

    launchable_name = "adv"

    default_params = {
        **Launchable.default_params,
        "attack" : ("CarliniWagner", {}),
        "batch_size" : 1,
    #     "prediction" : ("conf_file", None),
    #     "transform" : ("vae", {"experiment" : })
    }

    def create_id(self):

        prediction_conf, prediction_global_step = self._opts["prediction"]
        # name of last directory of the conf
        _id_pred = os.path.basename(os.path.dirname(prediction_conf))
        if prediction_global_step is not None:
            _id_pred += "-gs" + str(prediction_global_step)

        _id_transf = get_transform_id(*self._opts["transform"])

        _id = _id_pred + "/" + "r" + _id_transf + "/"

        _id += self.launchable_name

        #_id += "-" + get_attack_id(*opts["attack"])
        _id += get_attack_id(*self._opts["attack"])
        #n_images is not something which affects performances but simply visualization of figures..it is not mandatory to put in id..
        # _id += "-ni" + str(self._opts["n_images"])
        _id += "-as" + str(self._opts.get("n_samples_accuracy", 1))

        _id += '-bs' + str(self._opts['batch_size'])

        _id += "-ns" + str(self._opts["n_seeds"])

        super_id = super().create_id()
        _id += super_id
        return _id

    def __init__(self, opts, dirName, gpu=-1, seed=0, **unused):
        super().__init__(opts, dirName, seed)

        self._gpu = gpu

        self.sess = None

        tf.set_random_seed(seed)
        self.batch_size = opts["batch_size"]

        self._attack_tuple = opts["attack"]
        self._transform_tuple = opts["transform"]
        self._transform_model_id = get_transform_id(*self._transform_tuple) # transform create_id ??

        self._n_samples_acc = opts.get("n_samples_accuracy", 1)
        # import ipdb; ipdb.set_trace()
        self._transform_kwargs_accuracy = unpack_dict_of_lists(opts["transform_kwargs_accuracy"])
        if "transform_accuracy" in opts:
            self._transform_accuracy_list = opts["transform_accuracy"]["transf_list"]
            # self._transform_accuracy_model_id = get_transform_id(*self._transform_accuracy_tuple)
        else:
            self._transform_accuracy_list = None
        #self._model_type = opts["transform"][0]  #'ae' or 'vae'      THIS IS THE SAME AS self.transform_name

        self._attack_tuple = opts["attack"]
        self._prediction_conf, self._prediction_global_step = opts["prediction"]

        self._n_images = opts["n_images"]
        self._n_seeds = opts["n_seeds"]
        self._seeds = list(range(self._n_seeds))
        self._current_batch_size = self.batch_size

        # either epsilon is a list of interesting epsilons or it is the default list
        self._epsilons = opts['epsilons']['values']
        #opts.get('epsilons', [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
        self._epsilons_plus = [0.0] + self._epsilons

        # what is this??? Petru: the factor that doubles the epsilon if you go from image range [0,1] to [-1, 1], for example
        self._rescale_factor = 1.
        self._adversarial_dir = self.dirName + "/adversarial/"
        os.makedirs(self._adversarial_dir, exist_ok=True)
        self._accuracy_dir = self.dirName + "/accuracy/" + "d" # previously "def"
        self._stats_dir = self.dirName + '/statistics/'

        # put default values
        self._plot_reconstructions = opts["plot_adversarial_reconstructions"]
        self._only_plot_statistics = opts["only_plot_statistics"]
        self._do_plot_statistics = opts.get("plot_statistics", True)
        self._resize = opts["resize"]

        # TODO make this dependent on the transformation, not all transformation refers to external config!! Imagine random noise or jpeg compression
        # why scale is special? I could have a set of fields that I would like to see maybe?
        self._scale = "unimplemented"

        # # TODO get this in a cleaner way
        # if len(re.split('n[0-9]_s', self._transform_model_id)) > 1:
        #     self._scale = re.split('n[0-9]_s', self._transform_model_id)[1].split('-')[0]
        # elif len(re.split('n[0-9]_sm', self._transform_model_id)) > 1:
        #     self._scale = re.split('n[0-9]_sm', self._transform_model_id)[1].split('_')[0]
        # else:
        #     self._scale = 0.0

        self.attack_string = self.create_id().split('/')[-1] 


    def init(self, dataset):

        self.dataset = dataset

        self.create_feedable_placeholders()

        self.global_step = tf.train.get_or_create_global_step()

        self.create_input_nodes(dataset)

        self.create_session()

        self.create_network()
        self.prepare_attack()


    def create_input_nodes(self, dataset):
        """
        creates input nodes for a feedforward from the dataset

        Sets:
            x, y
        """

        datasets_nodes, handle, ds_initializers, ds_handles = self.create_datasets_with_handles(dataset)

        # perturbed dataset is not contemplated for the prediction case
        #if perturbed_dataset:
         #   raise Exception("perturbed datasets are not contemplated for the prediction case, use a regular dataset")

        #self.raw_x, self.raw_y = datasets_nodes
        
        self.ds_raw_x = datasets_nodes[0][0]
        self.ds_aug_x = datasets_nodes[0][1]
        self.ds_perturb_x = datasets_nodes[0][2]
        self.raw_x=self.ds_raw_x
        self.raw_y=datasets_nodes[1]
        self.x_shape_dict = {}
        self.x_shape_dict["train"] = dataset.x_shape_train
        self.x_shape_dict["eval"] = dataset.x_shape_eval

        self.x_shape = self.x_shape_dict["eval"]
        #import pdb;pdb.set_trace()
        self.x = self.raw_x
        self.x_tiled = tf.tile(self.x, [self._n_samples_acc]+ [1]*len(self.x_shape))

        self.x_adv = tf.placeholder(tf.float32, self.x.shape)
        self.x_adv_np = np.zeros((self._n_images,) + self.x_shape)

        self.y_shape = dataset.y_shape
        self.y = self.raw_y
        self.y_tiled = tf.tile(self.y, [self._n_samples_acc,])


    def create_datasets_with_handles(self, dataset):
        datasets_nodes, handle, ds_initializers, ds_handles = dataset.get_dataset_with_handle(
            self.batch_size, self.batch_size)
        self.datasets_initializers = ds_initializers
        self.datasets_handles_nodes = ds_handles
        self.ds_handle = handle
        self.raw_x = datasets_nodes[0]
        return datasets_nodes, handle, ds_initializers, ds_handles


    def create_feedable_placeholders(self):
        """
        DO NOT USE FOR MODEL SPECIFIC PLACEHOLDERS (e.g. losses or samples..)
        Create feedables. This function is setting global general purpose placeholders

        Sets:
            feedable placeholders with general purpose

        """
        self.is_training = False
        # self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

    def create_global_steps(self, n_points_train_set):
        self.global_step = tf.train.get_or_create_global_step()


    def create_network(self):
        """
        It gets the input nodes from the dataset and creates the network
        starting from the input nodes created by `create_input_nodes`

        Sets:
            network nodes depending on the specific child class
        """

        ffconffile = self._prediction_conf
        global_step_ff = self._prediction_global_step

        ff_dataset_conf, ff_model_parameters, ff_config = ArgoLauncher.process_conf_file(ffconffile)
        ff_dataset = Dataset.load_dataset(ff_dataset_conf)
        check_dataset_shapes(ff_dataset.x_shape_eval, self.x_shape)

        full_class_path = "prediction.core." + ff_model_parameters["model"]
        prediction_model_class = load_class(full_class_path)

        ff_network, ff_checkpoint_name = load_network(prediction_model_class, ffconffile, ff_dataset, global_step=global_step_ff)
        x_shape = (None,) + self.x_shape

        dummy_x = tf.placeholder(tf.float32, shape=x_shape, name='dummy_input')

        # LOAD FF NETWORK
        dummy_output = ff_network(dummy_x, is_training = self.is_training)
        ff_network.restore(self.sess, ff_checkpoint_name)

        # CALLABLE
        if isinstance(dummy_output, tfp.distributions.Distribution):
            def ff_module(inputs, is_training):
                return ff_network(inputs, is_training=is_training).logits
        else:
            ff_module = ff_network #.module


        self._ff_module = ff_module

        # CREATE TRANSFORM MODULES, one for attack with default params and one for accuracy calculation
        self.transform_name, self.transform_kwargs = self._transform_tuple
        self.transform_kwargs.update(
            {
                "dummy_x": dummy_x,
                "sess" : self.sess,
                "is_training" : self.is_training
            }
        )
        self._transform_module, self._transform_feedable = get_transform_module(self.transform_name, self.transform_kwargs)

        def _build_net(inputs):
            _x = self._transform_module(inputs)
            return self._ff_module(_x, is_training=self.is_training)

        self._build_net = _build_net

        #tile for better accuracy estimation
        self._logits = self._build_net(self.x_tiled)

        self._accuracy = 100. * tf.reduce_mean(tf.cast(
                           tf.equal(tf.argmax(self._logits, axis = 1),
                                 tf.cast(self.y_tiled, dtype = tf.int64)),
                                 dtype = tf.float32))

    def compute_accuracy_for_transf(self, inputs, labels, transf_tuple, **feed_kwargs):
        x_shape = (None,) + self.x_shape
        dummy_x = tf.placeholder(tf.float32, shape=x_shape, name='dummy_input')
        transform_name_accuracy, transform_kwargs_accuracy = transf_tuple
        transform_kwargs_accuracy.update(
            {
                "dummy_x": dummy_x,
                "sess" : self.sess,
                "is_training" : self.is_training
            }
        )
        transform_module_accuracy, transform_feedable_accuracy= get_transform_module(transform_name_accuracy, transform_kwargs_accuracy)
        logits = self._build_net(transform_module_accuracy(self.x_tiled))

        accuracy = 100. * tf.reduce_mean(tf.cast(
                        tf.equal(tf.argmax(logits, axis = 1),
                                tf.cast(self.y_tiled, dtype = tf.int64)),
                                dtype = tf.float32))

        return self.sess.run(accuracy, feed_dict={self.x: inputs,
                                                    self.y: labels,
                                                    **feed_kwargs})




    def prepare_attack(self):
        """
        It gets the input nodes from the dataset and creates the network
        starting from the input nodes created by `create_input_nodes`

        Sets:
            network nodes depending on the specific child class
        """


        input_shape = self.x_shape
        n_classes = self.dataset.n_labels
        attack_name, attack_kwargs = self._attack_tuple

        data_interval = [-1., 1.]

        # what is this rescale??
        # We save the rescaling factor in order to use it in the attack
        self._rescale_factor = data_interval[1] - data_interval[0]

        attack_kwargs.update(
            {
                "data_interval" : data_interval,
                "input_shape" : input_shape,
                "n_classes" : n_classes
            }
        )

        if attack_kwargs["proj_ord"] == "inf":
            attack_kwargs["proj_ord"] = np.inf

        if attack_kwargs.get("ldist_ord", None) == "inf":
            attack_kwargs["ldist_ord"] = np.inf

        self._attack_class = get_attack_class(attack_name)
        self._attack_kwargs = attack_kwargs



    def release(self):
        super().release()
        tf.reset_default_graph()
        self.sess.close()

    def create_session(self):
        # set some important options
        if self._gpu == -1:
            sess_config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)
        else:
            # config = tf.ConfigProto(log_device_placement=True)
            sess_config = tf.ConfigProto(allow_soft_placement=True)

        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config = sess_config)
        # TODO find a better way to do this
        self.sess.run(tf.variables_initializer([self.global_step]))
        self.datasets_handles = self.sess.run(self.datasets_handles_nodes)


    def attack(self, dataset_str):
        x_orig_np = self.dataset.get_raw_elements(dataset_str)[:self._n_images]
        y_orig_np = self.dataset.get_raw_labels(dataset_str)[:self._n_images]
        
        x_adv_shape = (self._n_images,) + self.x_shape

        for eps in self._epsilons:
            try:
                x_adv = self._load_np(dataset_str, eps)
                np.testing.assert_array_equal(x_adv.shape, x_adv_shape)

            except (IOError, AssertionError):
                self._current_batch_size = self.batch_size

                x_adv = np.zeros(x_adv_shape)

                attack_name, attack_kwargs = self._attack_tuple

                attack = self._attack_class(
                    self.sess,
                    self._build_net,
                    epsilon = self._rescale_factor * eps,
                    batch_size = self.batch_size,
                    **self._attack_kwargs
                )

                for i in tqdm(range(0, self._n_images, self.batch_size)):
                    if i + self.batch_size > self._n_images:
                        self._current_batch_size = self._n_images - i
                    x_adv[i:i+self._current_batch_size] = attack.run(x_orig_np[i:i+self._current_batch_size], y_orig_np[i:i+self._current_batch_size])

                self.x_adv_np = x_adv.astype('float32')
                self._save_np(dataset_str, eps, self.x_adv_np)

    def compute_accuracy(self, dataset_str, resize):
        # count also: eps = 0 corresponding to clean
        epsilons_plus = [0] + self._epsilons
        if self._transform_accuracy_list is None:
            for value_kwargs in self._transform_kwargs_accuracy:
                self.compute_accuracy_for_feeds(epsilons_plus, dataset_str, value_kwargs, resize)
        else:
            for transf in self._transform_accuracy_list:
                for value_kwargs in self._transform_kwargs_accuracy:
                    self.compute_accuracy_for_feeds(epsilons_plus, dataset_str, value_kwargs, resize, transf)


    def build_feed_dict(self, value_kwargs):
        nodes_kwargs = self._transform_feedable

        feed_dict = {nodes_kwargs[k] : value_kwargs[k] for k in nodes_kwargs}

        return feed_dict


    def compute_statistics(self, dataset_str):
        transform_tuple = (self.transform_name, self.transform_kwargs)
        os.makedirs(self._csv_stats_dir(transform_tuple), exist_ok=True)
        self.collect_stats_over_transf(dataset_str, transform_tuple)

    def collect_stats_over_transf(self, dataset_str, transform_tuple, param_specs=[('d',), ('stp',)]):
        ds_string = self.dirName.split('FF')[0].split('/')[-2]
        net_string = self._prediction_conf.split('/')[-2]
        transf_string = get_transform_id(*self._transform_tuple)
        attack_string = self.attack_string

        def spec_name(param_spec):
            return "_".join(param_spec)
        
        def get_field(string, field_spec):
            l = len(field_spec)
            if l==0 or l>2:
                raise ValueError("Not implemented tuple length `{:}`, found field spec `{:}`".format(l, field_spec))
            m = re.search('(-|^)' + field_spec[0] + '([\._A-Za-z0-9\,]+)' + '(-|$)', string)
            if m is None:
                ss1 = '0'
            else:
                ss1 = m.group(2)
            if l==1:
                return ss1
            m = re.search('(_|^)' + field_spec[1] + '([\.A-Za-z0-9\,]+)' + '(_|$)', ss1)
            if m is None:
                ss2 = '0'
            else:
                ss2 = m.group(2)
            return ss2

        
        def get_param_vals(string, param_specs):
            param_vals = []
            for ps in param_specs:
                pv = get_field(string, ps)
                param_vals.append(pv)
            return param_vals
        
        param_names = list(map(spec_name, param_specs))
        all_fields = ['eps', 'scale'] + param_names + ["adiff_avg", "adiff_std",
                                    "norm_avg", "norm_std",
                                    "norm_adv_avg", "norm_adv_std",
                                    "ndiff_avg", "ndiff_std",
                                    "nratio_avg", "nratio_std",
                                    "costh_avg", "costh_std",
                                    "costh_mu_avg", "costh_mu_std",
                                    "costh_mu_adv_avg", "costh_mu_adv_std",
                                    "cdist_avg", "cdist_std",
                                    "cdist_adv_avg", "cdist_adv_std",
                                    "diameter_mu",
                                    "diameter_mu_adv",]
        param_vals = [self._scale] + get_param_vals(transf_string, param_specs)
        base_dir = self.dirName.split('FF')[0].split(ds_string)[0]
        data = []
        for eps in self._epsilons:
            info_values_list = self.extract_one_point_info(base_dir, ds_string, net_string, get_transform_id(*transform_tuple), attack_string, dataset_str, eps)
            data.append([eps] + param_vals + info_values_list)
        df = pd.DataFrame(np.asarray(data), columns=all_fields)
        self._save_stats_df(dataset_str, transform_tuple, df)


    def compute_accuracy_for_feeds(self, epsilons_plus, dataset_str, value_kwargs, resize, transf_acc_tuple=None):
        full_transform_kwargs = {**self.transform_kwargs,
                                 **value_kwargs}

        if transf_acc_tuple is None:
            transform_tuple = (self.transform_name, full_transform_kwargs)
        else:
            transform_tuple = transf_acc_tuple
        if not resize:
            # import ipdb; ipdb.set_trace()
            os.makedirs(self._csv_dir(transform_tuple), exist_ok=True)
        else:
            os.makedirs(self._csv_dir_resize(transform_tuple), exist_ok=True)

        feed_kwargs = self.build_feed_dict(value_kwargs)

        x_orig_np = self.dataset.get_raw_elements(dataset_str)[:self._n_images]
        y_orig_np = self.dataset.get_raw_labels(dataset_str)[:self._n_images]

        n_epsilons_plus = len(epsilons_plus)

        accuracy_all = np.zeros((self._n_seeds, n_epsilons_plus))
        # avg_L2_norms_all = np.zeros((self._n_seeds, n_epsilons_plus))

        for s in self._seeds:
            tf.set_random_seed(s)

            for i in range(n_epsilons_plus):
                if i==0:
                    # clean
                    if(resize):
                        x_orig_tf = apply_resize(tf.constant(x_orig_np), self._resize["intermediate_size"])
                        x_orig_np = self.sess.run(x_orig_tf)
                    if transform_tuple is None:
                        accuracy_np = self.sess.run(self._accuracy, feed_dict={self.x: x_orig_np,
                                                                           self.y: y_orig_np,
                                                                           **feed_kwargs})
                    else:
                        accuracy_np = self.compute_accuracy_for_transf(x_orig_np, y_orig_np, transf_acc_tuple, **feed_kwargs)
                    accuracy_all[s, i] = accuracy_np

                else:
                    eps = epsilons_plus[i]
                    x_np = self._load_np(dataset_str, eps)

                    if(resize):
                        x_tf = apply_resize(tf.constant(x_np), self._resize["intermediate_size"])
                        x_np = self.sess.run(x_tf)
                    if transform_tuple is None:
                        accuracy_np = self.sess.run(self._accuracy, feed_dict={self.x: x_orig_np,
                                                                           self.y: y_orig_np,
                                                                           **feed_kwargs})
                    else:
                        accuracy_np = self.compute_accuracy_for_transf(x_orig_np, y_orig_np, transf_acc_tuple, **feed_kwargs)
                    accuracy_all[s, i] = accuracy_np

        acc_mean = np.mean(accuracy_all, axis=0)
        acc_std = np.std(accuracy_all, axis=0)

        df = pd.DataFrame({"eps": epsilons_plus, "acc_mean": acc_mean, "acc_std": acc_std},
                          columns = ["eps", "acc_mean", "acc_std"])
        if not resize:
            self._save_df(dataset_str, transform_tuple, df)
        else:
            self._save_df_resize(dataset_str, transform_tuple, df)


    def _save_np(self, dataset_str, eps, x_adv):
        np.save(self._npy_name(dataset_str, eps), x_adv)

    def _load_np(self, dataset_str, eps):
        return np.load(self._npy_name(dataset_str, eps))

    def _npy_name(self, dataset_str, eps):
        return self._adversarial_dir + dataset_str + '-eps' + str(eps) + '.npy'

    def _save_df(self, dataset_str, transform_tuple, df):
        df.to_csv(self._csv_name(dataset_str, transform_tuple), sep = " ", index = False)

    def _save_stats_df(self, dataset_str, transform_tuple, df):
        df.to_csv(self._csv_stats_name(dataset_str, transform_tuple), sep = " ", index = False)

    def _load_df(self, dataset_str, transform_tuple):
        return pd.read_csv(self._csv_name(dataset_str, transform_tuple), sep = " ")

    def _csv_name(self, dataset_str, transform_tuple):
         return self._csv_dir(transform_tuple) + "accuracy-" + dataset_str + '.csv'

    def _csv_stats_name(self, dataset_str, transform_tuple):
         return self._csv_stats_dir(transform_tuple) + "stats-" + dataset_str + '.csv'

    def _csv_dir(self, transform_tuple):
        transform_id = get_transform_id(*transform_tuple)
        return self._accuracy_dir + transform_id + "/"

    # TODO: make csv_dir generic
    def _csv_stats_dir(self, transform_tuple):
        return self._stats_dir + "/"

    def _save_df_resize(self, dataset_str, transform_tuple, df):
        df.to_csv(self._csv_name_resize(dataset_str, transform_tuple), sep = " ", index = False)

    def _csv_name_resize(self, dataset_str, transform_tuple):
         return self._csv_dir_resize(transform_tuple) + "accuracy-" + dataset_str + '.csv'

    def _csv_dir_resize(self, transform_tuple):
        transform_id = get_transform_id(*transform_tuple) + "-resize" + str(self._resize["intermediate_size"])
        return self._accuracy_dir + transform_id + "/"

    def plot_accuracies(self):
        #TODO what is meaning of this
        if self.transform_name in ['identity', 'ae']:
            df_list = glob(self.dirName + "/accuracy/*/*.csv")
        elif self.transform_name == 'vae':
            df_list = glob(self.dirName + "/accuracy/*1.0/*.csv")
        plt.figure(figsize=(20,10))
        self.plot_all(df_list)
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirName+"/accuracies.png", bbox_extra_artists=[lgd], bbox_inches='tight')
        plt.close()

    def plot_all(self, file_list):
        for filename in file_list:
            df = pd.read_csv(filename, sep = " ")
            plt.errorbar(df["eps"], df["acc_mean"], yerr=df["acc_std"],
                         linewidth=2, elinewidth=1, alpha=0.9,
                         label=self._label_from_filename(filename))

    def _label_from_filename(self, filename):
        tname, basename = filename.split("/")[-2:]
        dsname = os.path.splitext(basename)[0]
        return tname+"-"+dsname

    def generate_reconstructions(self, dataset_str):
        initial_images = []; rec_images = []

        for eps in [0.0] + self._epsilons:
            if(eps < 1e-4):  # i.e. eps == 0
                images_to_reconstruct = self.dataset.get_raw_elements(dataset_str)[self._plot_reconstructions["images_indexes"]]
            else:
                adv_ex_set = np.load(self._adversarial_dir + "test-eps" + str(eps) + ".npy")
                images_to_reconstruct = adv_ex_set[self._plot_reconstructions["images_indexes"]]

            reconstructions = self._transform_module(tf.constant(images_to_reconstruct))
            reconstructions_np = self.sess.run(reconstructions)

            initial_images.append(images_to_reconstruct)
            rec_images.append(reconstructions_np)
            #np.save(self._adversarial_dir + "rec_adv_eps" + str(eps) + ".npy", reconstructions_np)
        return initial_images, rec_images



    def plot_reconstructions(self, initial_images, rec_images):
        adversarial_reconstruction_folder = self.dirName + "/adversarial_reconstruction"
        os.makedirs(adversarial_reconstruction_folder, exist_ok=True)
        images_saver = ImagesSaver(adversarial_reconstruction_folder)

        rows = len(initial_images)
        panel = [[] for x in range(rows*2)]

        for i in range(0, rows):
            for j in range(len(self._plot_reconstructions["images_indexes"])): # include first and last image
                panel[2*i].append(initial_images[i][j])
                panel[2*i+1].append(rec_images[i][j])

                #panel[i+2].append(reconstructed_images[c])
                #if c == len(images)-1:
                #    break
                #else:
        # "[1st] interpolation in mu before sampling [2nd] iterpolation in z after sampling"
        images_saver.save_images(panel,
                                 fileName = "reconstructions", #+ str(ds_key) + "_" + self._time_reference_str + "_" + str(self._time_ref).zfill(4),
                                 title = "reconstructions 1) initial images 2) reconstructions \n", #+ self._plot_title,
                                 fontsize=9)


    def extract_latent_representation(self, transform_name):
        if(transform_name == 'ae'):
            mu = self._transform_module.encoder_module(self.x)
        elif(transform_name == 'vae'):
            mu = self._transform_module.encoder_module(self.x).mean()
        return mu

    def compute_latent_representation(self, dataset_str):
        # define node of interest
        mu = self.extract_latent_representation(self.transform_name)

        # get mu of clean; we need to know the latent size
        mu_clean_np = self.sess.run(mu, feed_dict={self.x: self.dataset.get_raw_elements(dataset_str)[:self._n_images],
                                                       })

        # define mu_np, where to store the mu of clean and adv. ex. and make mu clean its 0th elem.
        mu_np = np.zeros((len(self._epsilons_plus), self._n_images, mu_clean_np.shape[1]))
        mu_np[0] = mu_clean_np

        # get mu adv for all eps
        for i in range(1, len(self._epsilons_plus)):
            eps = self._epsilons_plus[i]
            x_np = self._load_np(dataset_str, eps)
            mu_np[i] = self.sess.run(mu, feed_dict={self.x: x_np,
                                                                   })
        return mu_np

    def compute_latent_representation_for_eps(self, x, eps):
        x_ph = tf.placeholder(tf.float32, shape=x.shape)
        if(self.transform_name == 'ae'):
            mu_star = self._transform_module.encoder_module(x_ph)
        elif(self.transform_name == 'vae'):
            mu_star = self._transform_module.encoder_module(x_ph).mean()
        mu_star_np = self.sess.run(mu_star, feed_dict={x_ph: x[:self._n_images]})
        return mu_star_np


    def extract_one_point_info(self, base_dir, ds_string, net_string, transf_string, attack_string, dataset_str, eps):

        def rough_sqdist_numerics_diagzero(sqdistances):
            np.fill_diagonal(sqdistances, 0.)
            min_dist = np.min(sqdistances)
            max_dist = np.max(sqdistances)
            if min_dist < 0.:
                sqdistances += -np.min(sqdistances)
                np.fill_diagonal(sqdistances, 0.)
            return sqdistances

        def riemannian_dist(xp, xq, g_matrix):
            Ixp = np.matmul(g_matrix, xp.T)
            sqnormp = np.sum(xp * Ixp.T, axis=1)
            Ixq = np.matmul(g_matrix, xq.T)
            sqnormq = np.sum(xq * Ixq.T, axis=1)
            sqdistances = sqnormp.reshape(-1, 1) + sqnormq.reshape(1, -1) - 2 * np.matmul(xp, Ixq)
            # # temporarily remove this check put a rough compliance function
            # sqdistances = check_sqdist_numerics_diagzero(sqdistances)
            sqdistances = rough_sqdist_numerics_diagzero(sqdistances)
            distances = np.sqrt(sqdistances)
            return distances

        x_adv = np.load(os.path.join(*[self.dirName, "adversarial", dataset_str+"-eps{:}.npy".format(eps)]))
        x_clean = self.dataset.get_raw_elements(dataset_str)[:self._n_images]

        x_ph = tf.placeholder(tf.float32)

        mu_np = self.compute_latent_representation_for_eps(x_clean, eps)
        mu_star_np = self.compute_latent_representation_for_eps(x_adv, eps)
        norm_np = np.linalg.norm(mu_np, axis=1)
        norm_star_np = np.linalg.norm(mu_star_np, axis=1)
        adiff = norm_star_np - norm_np
        costh = np.sum(mu_np * mu_star_np, axis=1)/(norm_np*norm_star_np)
        
        onevec = np.ones((1, mu_np.shape[1]))
        norm_onevec = np.linalg.norm(onevec)
        costh_mu = np.sum(mu_np * onevec, axis=1)/(norm_np*norm_onevec)
        costh_mu_adv = np.sum(mu_star_np * onevec, axis=1)/(norm_star_np*norm_onevec)
        nratio = norm_star_np/norm_np
        
        ldim = mu_np.shape[1]
        euclidean_dist = partial(riemannian_dist, g_matrix=np.eye(ldim))
        
        distances_mu_np = euclidean_dist(mu_np, mu_np)
        distances_mu_star_np = euclidean_dist(mu_star_np, mu_star_np)
        diameter_mu_set = np.max(distances_mu_np)
        diameter_mu_star_set = np.max(distances_mu_star_np)
        
        centroid_mu_set = np.mean(mu_np, axis=0)
        centroid_mu_star_set = np.mean(mu_star_np, axis=0)
        
        cdists_set = euclidean_dist(mu_np, centroid_mu_set.reshape(1, -1))
        cdists_adv_set = euclidean_dist(mu_star_np, centroid_mu_star_set.reshape(1, -1))
        
        ndiffs_np = np.linalg.norm(mu_star_np-mu_np, axis=1)
        ndiff_avg = np.mean(ndiffs_np)
        ndiff_std = np.std(ndiffs_np)
        
        adiff_avg = np.mean(adiff)
        adiff_std = np.std(adiff)
        nratio_avg = np.mean(nratio)
        nratio_std = np.std(nratio)
        costh_avg = np.mean(costh)
        costh_std = np.std(costh)
        costh_mu_avg = np.mean(costh_mu)
        costh_mu_std = np.std(costh_mu)
        costh_mu_adv_avg = np.mean(costh_mu_adv)
        costh_mu_adv_std = np.std(costh_mu_adv)

        norms_avg = np.mean(norm_np)
        norms_std = np.std(norm_np)
        
        norms_adv_avg = np.mean(norm_star_np)
        norms_adv_std = np.std(norm_star_np)

        cdists_set_mean = np.mean(cdists_set)
        cdists_set_std = np.std(cdists_set)
        cdists_adv_set_mean = np.mean(cdists_adv_set)
        cdists_adv_set_std = np.std(cdists_adv_set)
        
        return [adiff_avg, adiff_std,
                norms_avg, norms_std,
                norms_adv_avg, norms_adv_std,
                ndiff_avg, ndiff_std,
                nratio_avg, nratio_std,
                costh_avg, costh_std,
                costh_mu_avg, costh_mu_std,
                costh_mu_adv_avg, costh_mu_adv_std,
                cdists_set_mean, cdists_set_std,
                cdists_adv_set_mean, cdists_adv_set_std,
                diameter_mu_set,
                diameter_mu_star_set,
            ]

    def plot_mu_statistics(self, dataset_str):
        mu_np = self.compute_latent_representation(dataset_str)

        mu_statistics_folder = self.dirName + "/mu_statistics"
        os.makedirs(mu_statistics_folder, exist_ok=True)

        self.plot_errorbars_average_norm(dataset_str, mu_np, mu_statistics_folder)
        self.plot_histogram_norm_of_difference(dataset_str, mu_np, mu_statistics_folder)
        self.plot_histogram_difference_of_norms(dataset_str, mu_np, mu_statistics_folder)
        self.plot_histogram_cos_adv_clean(dataset_str, mu_np, mu_statistics_folder)
        self.plot_histograms_norm_per_images(dataset_str, mu_np, mu_statistics_folder, no_images = 10)
        self.plot_histogram_norms(dataset_str, mu_np, mu_statistics_folder)


    def plot_errorbars_average_norm(self, dataset_str, mu, folder_name):
        all_norms = np.linalg.norm(mu, axis = 2)
        average_norms = np.average(all_norms, axis = 1)
        std_norms = np.std(all_norms, axis = 1)

        plt.errorbar(self._epsilons_plus, average_norms, std_norms, capsize = 5)
        plt.title("Average norm of latent representation")
        plt.savefig(folder_name + "/mu_average_norms_errorbars.png")
        plt.close()

        df = pd.DataFrame({"eps": self._epsilons_plus, "avg_norms": average_norms, "avg_norms_std": std_norms},
                          columns = ["eps", "avg_norms", "avg_norms_std"])
        df.to_csv(folder_name + '/average_norms_text.csv', sep = " ", index = False)


    def plot_histogram_norm_of_difference(self, dataset_str, mu, folder_name):
        mu_diff_adv_clean = [mu[i] - mu[0] for i in range(1, len(self._epsilons_plus))]
        mu_diff_norms = np.linalg.norm(mu_diff_adv_clean, axis = 2)

        bins = np.linspace(0, 15, num = 51)
        for i in range(len(self._epsilons_plus)-1):
            plt.hist(mu_diff_norms[i], bins, color = 'b')
            plt.title("||mu_adv - mu_clean||, eps = " + str(self._epsilons_plus[i+1]))
            plt.xlabel('Norm')
            plt.ylabel('No. images')
            plt.text(0.1, 0.3, self._transform_model_id)
            plt.savefig(folder_name + "/hist_norms_of_diff_" + str(self._epsilons_plus[i+1]) + ".png")
            plt.close()


    def plot_histogram_difference_of_norms(self, dataset_str, mu, folder_name):
        mu_norms = np.linalg.norm(mu, axis = 2)
        mu_norms_diff_adv_clean = [mu_norms[i] - mu_norms[0] for i in range(1, len(self._epsilons_plus))]

        bins = np.linspace(-7, 7, num = 51)
        for i in range(len(self._epsilons_plus)-1):
            plt.hist(mu_norms_diff_adv_clean[i], bins, color = 'm')
            plt.title("||mu_adv|| - ||mu_clean||, eps = " + str(self._epsilons_plus[i+1]))
            plt.xlabel('Difference')
            plt.ylabel('No. images')
            plt.text(0.1, 0.3, self._transform_model_id)
            plt.savefig(folder_name + "/hist_diff_of_norms_" + str(self._epsilons_plus[i+1]) + ".png")
            plt.close()


    def plot_histogram_cos_adv_clean(self, dataset_str, mu, folder_name):
        inner_prod_adv_clean = [np.diag(np.matmul(mu[0], np.transpose(mu[i]))) for i in range(1, len(self._epsilons_plus))]
        prod_norm_adv_clean = [np.multiply(np.linalg.norm(mu[0], axis = 1), np.linalg.norm(mu[i], axis = 1)) for i in range(1, len(self._epsilons_plus))]
        cos_adv_clean = [np.divide(inner_prod_adv_clean[i], prod_norm_adv_clean[i]) for i in range(len(self._epsilons_plus)-1)]

        bins = np.linspace(-1, 1, num = 51)
        for i in range(len(self._epsilons_plus)-1):
            plt.hist(cos_adv_clean[i], bins, color = 'g')
            plt.title("cos(adv, clean), eps = " + str(self._epsilons_plus[i+1]))
            plt.xlabel('Cos angle')
            plt.ylabel('No. images')
            plt.text(0.1, 0.3, self._transform_model_id)
            plt.savefig(folder_name + "/hist_cos_adv_clean_" + str(self._epsilons_plus[i+1]) + ".png")
            plt.close()


    def plot_histograms_norm_per_images(self, dataset_str, mu, folder_name, no_images):
        mu_norms = np.linalg.norm(mu[:, :10,:], axis = 2)

        colors = ['m', 'b', 'g', 'r', 'y', 'c', 'k', '0.5', '#C40000', '#006400']
        for i in range(no_images):
            plt.plot(self._epsilons_plus, mu_norms[:,i], color = colors[i])
        plt.title("Norm of mu, individual images")
        plt.xlabel('Epsilon')
        plt.ylabel('Norm')
        plt.text(0.1, 0.3, self._transform_model_id)
        plt.savefig(folder_name + "/graph_norm_individual_images.png")
        plt.close()
        
    # Turns floats of the form a.bc... into "a-bc...", so that the plots can be put on Overleaf
#     def latex_friendly(x):
#         char_x = str(x)
#         new_char_x = char_x[0] + '-' + char_x[2:]
#         return new_char_x
        

    def plot_histogram_norms(self, dataset_str, mu, folder_name):
        #import pdb;pdb.set_trace()
        mu_norms = np.linalg.norm(mu, axis = 2)
        
        bins = np.linspace(0, 20, num = 51)
        for i in range(len(self._epsilons_plus)):
            plt.hist(mu_norms[i], bins, color = 'm')
            plt.title("Norms of latent representations, eps = " + str(self._epsilons_plus[i]))
            plt.xlabel('L2 Norm')
            plt.ylabel('No. images')
            plt.text(0.1, 0.3, self._transform_model_id)
            plt.savefig(folder_name + "/hist_norms_" + str(self._epsilons_plus[i]) + ".png")
            plt.close()
        
         

