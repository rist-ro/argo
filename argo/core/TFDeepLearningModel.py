import os
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from datasets.Dataset import Dataset, TRAIN_LOOP
from .ArgoLauncher import ArgoLauncher
from .DeepLearningModel import DeepLearningModel
from .hooks.ArgoHook import STEPS, EPOCHS
from .hooks.LoggingMeanTensorsHook import LoggingMeanTensorsHook
from .hooks.CheckpointSaverHook import CheckpointSaverHook
from .hooks.ImagesInputHook import ImagesInputHook
from .hooks.FisherMatrixHook import FisherMatrixHook

from .Regularizers import Regularizers

from .optimizers.NaturalGradientOptimizer import NaturalGradientOptimizer

from .argoLogging import get_logger
from .utils.argo_utils import AC_REGULARIZATION, load_class, load_module, get_clipping_id, eval_method_from_tuple, \
    NUMTOL, CUSTOM_REGULARIZATION

from itertools import chain

import importlib

import pdb

tf_logging = get_logger()

from .optimizers.TFOptimizers import TFOptimizers

def load_model(conf_file, global_step=None, dataset=None, gpu=0, seed=0, model_class_base_path='', monitorSession=True):
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
    model.init(dataset)
    # network = model._network
    # network.init_saver()
    
    model.create_session(model_parameters, config, monitorSession=monitorSession)

    #if global_step is None it will restore the last checkpoint in the folder model._checkpoint_dir, you can pass global_step to restore a particular chackpoint
    model.restore(global_step = global_step)

    # if save_net:
    #     sess = model.get_raw_session()
    #
    #     if network_dir:
    #         path = network_dir+"/"+get_full_id(dataset, model)+'/'+network.name
    #     else:
    #         path = model.dirName+'/networks/'+network.name
    #
    #     network.save(sess, path, global_step=global_step)

    return model, dataset


def load_model_without_session(conf_file, global_step=None, dataset=None, gpu=0, seed=0, model_class_base_path=''):
    """Load a TFDeepLearningModel without session

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
    model.init(dataset)

    # network = model._network
    # network.init_saver()
    
    checkpoint_name = model.checkpoint_name(global_step)

    return model, dataset, checkpoint_name


def load_network(ArgoTFDeepLearningModelClass, conf_file, dataset, global_step=None):
    """Load the network of a specific model and the corresponding checkpoint.
    The Network needs to be applied (to generate the variables, that are instantiated in the _build of Sonnet)
    and then restored from the checkpoint.

    e.g.
    ```
    network, checkpoint_name = load_network(ClassificationModel, model_dir,
                                        dataset, model_params, config)
    logits = network(x)
    network.restore(sess, checkpoint_name)
    ```

    Args:
        ArgoTFDeepLearningModelClass (Class): the TFDeepLearningModel class to load.
        conf_file (str): the conf file of the model where to find the experiment.
        dataset (datasets.Dataset): (optional) the argo Dataset of the model for the training. If not passed it will be reloaded.
        global_step (int): (optional) the global step to load the checkpoint (if None the last checkpoint present will be loaded).

    Returns:
        ArgoAbstractNetwork: the Argo Network to load
        str: checkpoint_name
    """

    dataset_conf, model_parameters, config = ArgoLauncher.process_conf_file(conf_file)
    # parallelism = 0 # 0 is equivalent to single

    # if dataset is not None:
    #     print("load_network: `dataset` IS DEPRECATED, will be removed soon")
    # if model_class_base_path is not '':
    #     print("load_network: `model_class_base_path` IS DEPRECATED, will be removed soon")
    #

    update_model_params(model_parameters, dataset)

    model_dir = os.path.split(os.path.dirname(conf_file))[0]
    model = ArgoTFDeepLearningModelClass(model_parameters, model_dir)

    network = model._network
    checkpoint_name = model.checkpoint_name(global_step)
    return network, checkpoint_name


def update_model_params(model_parameters, dataset):
    try:
        output_shape = dataset.y_shape
    except ValueError:
        output_shape = None

    dataset_info = {"output_shape": output_shape,
                    "input_shape": dataset.x_shape_train}

    model_parameters.update(dataset_info)


class TFDeepLearningModel(DeepLearningModel):

    default_params= {
        **DeepLearningModel.default_params,

        "stochastic" : 1, # add Gaussian noise to the dataset
        "stochastic_noise_param" : 0, # variance of the noise added to the dataset

        #TODO never rescale
        # "rescale" : 0., # rescale inputs in [rescale, 1-rescale] (for numerical issues)

        "optimizer": ("AdamOptimizer", {"learning_rate" : 0.001,
                                        "beta1": 0.9,
                                        "beta2":0.999}),

        "regularizers" : {},
        
        "grad_clipping" : (None, {}),
        #"natural_gradient" : [0],

        "batch_size_train" : 128,
        "batch_size_eval" : 512,
    }

    def create_id(self):

        _id = '-st' + str(self._opts["stochastic"]) +\
              '-stp' + str(self._opts["stochastic_noise_param"]) + \
              '-bs' + str(self._opts["batch_size_train"]) + \
              '-tr' + TFOptimizers.create_id(self._opts["optimizer"]) + \
              '-c' + get_clipping_id(self._opts["grad_clipping"])              

        if "note" in self._opts:
            _id += '-N' + self._opts["note"]
        if "pretrained_checkpoint" in self._opts:
            longname = os.path.abspath(self._opts["pretrained_checkpoint"])
            shortname = "".join([i[0] for i in longname.split("/")[1:]])
            _id += '-PC' + shortname
        super_id = super().create_id()
        _id += super_id
        return _id

    def __init__(self, opts, dirName, check_ops = False, gpu=-1, seed=0):

        # this needs to be called before the parent constructor
        #self._regularizers = opts.get("regularizers", {})
        #pdb.set_trace()
        
        super().__init__(opts, dirName, seed)
        
        # moved up the hierarchy
        # self.dirName = dirName + "/" + self._id

        self._check_ops = check_ops
        self._numerics_ops = None

        self._gpu = gpu

        self.sess = None
        self._saver = None
        self.global_step = None

        tf.set_random_seed(seed)

        #restore checkpoint
        self._restore_chkptfile = (None if "pretrained_checkpoint" not in self._opts else self._opts["pretrained_checkpoint"])
        #checkpoints
        self._checkpoint_dir = self.dirName + "/saved_models/"
        #tensorboard
        self._tensorboard_dir = self.dirName + "/tensorboard/"

        self.summary_keys = [tf.GraphKeys.SUMMARIES]
        self.summary_nodes = {ck:[] for ck in self.summary_keys}
        self.summary_writers = {ck:[] for ck in self.summary_keys}

        self.stochastic = self._opts["stochastic"]
        if self.stochastic==0:
            #no noise is added
            pass
        elif self.stochastic == 1:
            self._clip_after_noise = True
        elif self.stochastic== 2 :
            self._clip_after_noise = False

        else:
            raise ValueError("stochastic can be: 0 no noise, 1 noise and clip after,"
                             "2 noise and do not clip after. Found stochastic {}".format(self.stochastic))

        self.stochastic_noise_param = self._opts["stochastic_noise_param"]

        # # TODO never rescale
        if "rescale" in self._opts:
            raise KeyError("the key `rescale` is not supported anymore. Rescaling is not allowed, remove it from the conf.")
        # self.rescale = opts["rescale"] # rescale the inputs if they are continuous

        self.batch_size = {}
        self.batch_size["train"] = self._opts["batch_size_train"]
        self.batch_size["eval"] = self._opts["batch_size_eval"]

        # important nodes
        self.x = None
        self.y = None
        self.x_shape = {}

        self.optimizer_tuple = self._opts["optimizer"]
        #self.compute_natural_gradient = opts["natural_gradient"]

        #self.learning_rate = opts["training_algorithm"]["learning_rate"]

        #self.training_algorithm = opts["training_algorithm"]["algorithm"]
        #self.learning_rate = opts["training_algorithm"]["learning_rate"]

        self._grad_clipping_tuple = self._opts["grad_clipping"]

        # important nodes
        self.loss = None
        self.regularizers = []
        
        # create regularizers
        if ("regularizers" not in self._opts) or ("weights" in self._opts["regularizers"].keys() or "bias" in self._opts["regularizers"].keys() or "custom" in self._opts["regularizers"].keys()) or len(self._opts["regularizers"].keys())==0:
            self.custom_regularizers = []
        else:
            self.custom_regularizers = {}
            for key in self._opts["regularizers"].keys():
                self.custom_regularizers[key] = []
        
        self.update_ops = []
        # list of kl_losses on the weights in case of bayesian learning
        self.kl_losses = []

        self.datasets_initializers = {}
        self.datasets_handles_nodes = {}
        self.datasets_handles = {}

        # passed to ChechpoitSaverHook
        self._pb_output_nodes = None

    # NB dataset can be none in certain contexts,
    # NB maybe there is a better way to do this (Luigi)
    # TODO YES, this function should take the input node
    # IF using a CNN, shape of input node Must be of len 4, both sonnet functions and tf.layers.conv2d require this convention and this makes the code uniform: input_shape = [None, size1, size2, channels]
    # TODO-ARGO2 this function should restore variables from the last checkpoint if it is present, not always initialize the variables
    def init(self, dataset):

        self.binary = dataset.binary_input

        #TODO these two are probably useless... if you need the input shape just do tf.shape(self.raw_x) for some networks the input could change from train to eval
        #TODO if there is a way to avoid using explicitly the input dimension it is probably better...
        self.x_shape["train"] = dataset.x_shape_train
        self.x_shape["eval"] = dataset.x_shape_eval

        self.dataset = dataset

        self.create_feedable_placeholders()

        # create global steps
        self.create_global_steps(dataset.n_samples_train)

        self.create_input_nodes(dataset)

        # set optimizer
        self.set_optimizer()

        #self.create_is_training_node()

        self.create_network()

        #TODO-ARGO2 create loss, regularizer, optimizer and global step are typical of a training algorithm
        #TODO-ARGO2 how do we handle other cases that do not have a train? I think we should define
        #TODO-ARGO2 TrainableTFDeepLearningModel and redefine init

        # define self.loss and check it is finite
        self.create_loss()

        self.create_custom_regularizers()
                
        # define self.regularizers and self.update_ops
        self.create_regularizers_and_updates()

        # set the training operation for self.loss + self.regularizers + self.custom_regularizers
        self.set_training_op()
        
        # not used at the moment, could be useful at a certain point
        # self.create_random_update_op()

        # there are case in which multiple losses exit
        if isinstance(self.loss, dict):
            for k, v in self.loss.items():
                self.loss[k] = tf.check_numerics(v, "self.loss" + str(k) + " is not finite")
        else:
            self.loss = tf.check_numerics(self.loss, "self.loss is not finite")

        #session will be created after init
    
        
    def create_datasets_with_handles(self, dataset):
        datasets_nodes, handle, ds_initializers, ds_handles = dataset.get_dataset_with_handle(self.batch_size["train"], self.batch_size["eval"])
        self.datasets_initializers = ds_initializers
        self.datasets_handles_nodes = ds_handles
        self.ds_handle = handle
        self.datasets_nodes = datasets_nodes # this is needed, since ds_raw_x may be modified in create_input_nodes to remove the mask
        
        self.ds_raw_x = datasets_nodes[0][0]
        self.ds_aug_x = datasets_nodes[0][1]
        self.ds_perturb_x = datasets_nodes[0][2]
        
        return datasets_nodes, handle, ds_initializers, ds_handles

    def create_feedable_placeholders(self):
        """
        DO NOT USE FOR MODEL SPECIFIC PLACEHOLDERS (e.g. losses or samples..)
        Create feedables. This function is setting additional placeholder
        (it probably should never be used since placeholders should be set 3in the right places)

        Sets:
            feedable placeholders with general purpose

        """

        self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

    #def create_is_training_node(self):
    #    self._is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

    @abstractmethod
    def create_network(self):
        """
        It gets the input nodes from the dataset and creates the network
        starting from the input nodes created by `create_input_nodes`

        Sets:
            network nodes depending on the specific child class
        """
        pass

    @abstractmethod
    def create_input_nodes(self, dataset):
        """
        create input nodes for the network
        starting from the dataset

        Sets:
            input nodes depending on the specific child class
        """
        pass

    @abstractmethod
    def create_loss(self):
        """create loss nodes for the network
        based on the nodes that create_networks has created,
        this method will create the loss nodes

        Sets:
            self.loss
            other additional loss nodes to be monitored during train can be set

        """
        pass

    # create custom regularizers    
    def create_custom_regularizers(self):
        
        if isinstance(self.custom_regularizers, list):
            self._create_custom_regularizers()
        elif isinstance(self.custom_regularizers, dict):
            for key in self.custom_regularizers.keys():
                # add regularizers for discriminator
                self._create_custom_regularizers(key)
        else:
            raise Exception("self.custom_regularizers should be a list or a dict")

    def _create_custom_regularizers(self, network=None):
        if network is None:
            regularizers = self._opts["regularizers"]
            custom_regularizers = self.custom_regularizers
        else:
            regularizers = self._opts["regularizers"][network]
            custom_regularizers = self.custom_regularizers[network]

        '''
        if "custom" in regularizers.keys():
            
            for regularizer_tuple in regularizers["custom"]:

                regularizer_name = regularizer_tuple[0]
                regularizer_tuple[1]["model"] = self
                custom_regularizer = Regularizers.instantiate_regularizer(regularizer_tuple, module_path = "")
                ''
                regularizer_name = regularizer_tuple[0]
                regularizer_kwargs = regularizer_tuple[1]
                regularizer_kwargs["model"] = self
                
                try:
                    # load customized regularizers from core/Regularizers.py
                    reg_module = importlib.import_module("core.Regularizers", '.'.join(__name__.split('.')[:-1]))
                    custom_regularizer, _, _ = eval_method_from_tuple(reg_module, (regularizer_name, regularizer_kwargs))
                except AttributeError as e:
                    # try to load from argo
                    try:
                        # load customized regularizers from core/argo/core/Regularizers.py
                        reg_module = importlib.import_module(".Regularizers", '.'.join(__name__.split('.')[:-1]))
                        custom_regularizer, _, _ = eval_method_from_tuple(reg_module, (regularizer_name, regularizer_kwargs))

                    except AttributeError as e:
                        raise AttributeError("regularizer %s not found" % regularizer_name) from e
                ''

                custom_regularizers.append(custom_regularizer)
                self.check_regularizers(regularizer_name, network)
        '''

        if "custom" in regularizers.keys():
                    
            for regularizer_tuple in regularizers["custom"]:

                regularizer_name = regularizer_tuple[0]
                regularizer_tuple[1]["model"] = self
                custom_regularizer = Regularizers.instantiate_regularizer(regularizer_tuple, module_path = "")

                custom_regularizers.append(custom_regularizer)
                self.check_regularizers(regularizer_name, network)

                

    def check_regularizers(self, regularizer_name, network=None):
        pass

    '''
    def create_custom_regularizers(self):
        # should not be an empty list
        return [0.]
    '''

    # save in self.regularizers the regularizers of the model
    def create_regularizers_and_updates(self):

        wb_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # see keras_utils.py: activity_and_contractive_regularizers
        ac_regularizers = tf.get_collection(AC_REGULARIZATION)
        # if (not wb_regularizers) and (not ac_regularizers):
        #     wb_regularizers = [tf.constant(0.)]

        #import pdb;pdb.set_trace()
        if len(wb_regularizers)>0:
            self.regularizers += wb_regularizers
        if len(ac_regularizers)>0:
            self.regularizers += ac_regularizers

        # self.regularizers += ([self.custom_regularizers[r] for r in self._opts["regularizers"].keys() if len(self.custom_regularizers[r])>0])
        # we need to flatten the list if we have both custom regularizers and another type of regularizers
        # (weight/bias or contractive)
        self.regularizers += list(chain.from_iterable([self.custom_regularizers[r]
                                                       for r in self._opts["regularizers"].keys()
                                                       if len(self.custom_regularizers[r]) > 0]))

        self.update_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # ac_train_regularizers = tf.get_collection(get_ac_collection_name("train"))
        # ac_validation_regularizers = tf.get_collection(get_ac_collection_name("validation"))
        # ac_test_regularizers = tf.get_collection(get_ac_collection_name("test"))
        # self.regularizer["train"] = tf.add_n(wb_regularizers + ac_train_regularizers, name="regularization_train")
        # self.regularizer["validation"] = tf.add_n(wb_regularizers + ac_validation_regularizers, name="regularization_validation")
        # self.regularizer["test"] = tf.add_n(wb_regularizers + ac_test_regularizers, name="regularization_test")

    
    def create_global_steps(self, n_points_train_set):
        self.n_batches_per_epoch = np.ceil(n_points_train_set/self.batch_size["train"])

        self.global_step = tf.train.get_or_create_global_step()
        self.global_epoch = tf.cast(tf.floor(tf.cast(self.global_step, tf.float32) /
                                            self.n_batches_per_epoch),
                                    tf.int64, "global_epoch")

        tf.add_to_collection("global_epoch", self.global_epoch)

    # this creates an operation to add to all trainable variables a white noise of param
    # std = tf.sqrt(variance)/10
    def create_random_update_op(self):

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        update_opts = []
        for var in vars:

            _, variance = tf.nn.moments(tf.reshape(var,[-1]),axes=[0])

            normal = tf.distributions.Normal(loc=0.0, scale=tf.sqrt(variance)/10)
            white_noise = normal.sample(var.get_shape())

            update_opts.append(var.assign(var + white_noise))

        self.random_update_op = tf.group(update_opts)

    #apply clipping
    def _clip_gradients(self, grads_and_vars, grad_clipping_tuple):

        clipping_method, clipping_kwargs = grad_clipping_tuple

        grads_and_vars_not_none = [(g, v) for (g, v) in grads_and_vars if g is not None]
        grads = [g for (g, v) in grads_and_vars_not_none]
        variables = [v for (g, v) in grads_and_vars_not_none]

        #self.grad_norms = [tf.norm(g) for g in grads]

        #else:
        #    self.new_logits = self.logits

        self.grads = grads
        self.grads_norm = tf.global_norm(grads)

        #see https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
        #if clipping_method == "clip_by_global_norm":
        #
        #    print_ops = [tf.print("global norm " + str(tf.norm(g))) for g in grads]
        #    with tf.control_dependencies(print_ops):
        #        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clipping_kwargs["value"])
        #    self.clipped_grads_and_vars = [(clipped_grads[i], variables[i]) for i in range(len(grads))]


        #see https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
        if clipping_method == "clip_by_global_norm":

            #clip_by_global_norm requires all the grads as argument, not only grad[i]
            grads_and_vars_not_none = [(g, v) for (g, v) in grads_and_vars if g is not None]
            grads = [g for (g, v) in grads_and_vars_not_none]
            variables = [v for (g, v) in grads_and_vars_not_none]

            #print_ops = [tf.print("loss=", self.loss)] + [tf.print("norms_g", tf.norm(g)) for g in grads] + [tf.print("g", g) for g in grads] + [tf.print("p", self.prob)] + [tf.print("invFisher", self.invFisher)] + [tf.print("invA", self.invA)] + [tf.print("invB", self.invB)]

            #print_ops = [tf.print("partA =", self.partA, summarize=-1), tf.print("partB =", self.partB, summarize=-1), tf.print("prob_sliced =", self.prob_sliced, summarize=-1), tf.print("natural_gradient_theta =", self.natural_gradient_loss_theta, summarize=-1), tf.print("euclidean gradient =", self.euclidean_gradient, summarize=-1), tf.print("TA =", self.TA, summarize=-1), tf.print("Tup =", self.Tup, summarize=-1) , tf.print("Tdown =", self.Tdown, summarize=-1) , tf.print("TB =", self.TB, summarize=-1)] + [tf.print("norms_g", tf.norm(g), summarize=-1) for g in grads]
            #with tf.control_dependencies(print_ops):
            clip_value = clipping_kwargs["value"]
            clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_value)
            clipped_grads_and_vars = [(clipped_grads[i], variables[i]) for i in range(len(grads))]

        elif clipping_method == "clip_by_norm":

            grads_and_vars_not_none = [(g, v) for (g, v) in grads_and_vars if g is not None]

            grads = [g for (g, v) in grads_and_vars_not_none]
            variables = [v for (g, v) in grads_and_vars_not_none]

            # How t handle numerical issues
            # 1) set nan/inf to zero
            # grads = [tf.where(tf.is_finite(g), g, tf.zeros_like(g)) for (g, v) in grads_and_vars_not_none]
            # 2) set nan/inf to noisy gradient,
            #grads = [tf.where(tf.is_finite(g), g, tfd.Normal(loc=0.0, scale=tf.sqrt(tf.nn.moments(tf.reshape(v,[-1]),axes=[0])[1])/10 + 0.01).sample(g.get_shape())) for (g, v) in grads_and_vars_not_none]

            clip_value = clipping_kwargs["value"]
            clipped_grads_and_vars = [(tf.clip_by_norm(g, clip_value), v) for (g, v) in zip(grads, variables)]

        elif clipping_method == "clip_by_value":

            clip_value = clipping_kwargs["value"]
            clipped_grads_and_vars = [(tf.clip_by_value(g, -clip_value, clip_value), v) for (g, v) in grads_and_vars if g is not None]

        elif not clipping_method:

            grads_and_vars_not_none = [(g, v) for (g, v) in grads_and_vars if g is not None]
            clipped_grads_and_vars = grads_and_vars_not_none

        else:
            raise Exception("clipping method not recognized: " + clipping_method)

        return clipped_grads_and_vars

    ''''not used, please don't delete (Luigi)
    def check_numerics_grad(grads):

        condition = tf.constant(True)
        for g in grads:
            condition = tf.math.logical_and(condition,tf.check_numerics(g, "check numerics for gradient failed"))
        return condition

        # the function above could be used in the following context
        self.clipped_grads_and_vars = tf.cond(check_numerics_grad(grads),
                                              self.clip_gradients(self.grads_and_vars, self._grad_clipping_tuple),
                                              grads_and_vars_not_none)
                                              #grads_and_vars_not_none)

    '''

    def set_optimizer(self):

        with tf.variable_scope('optimizer'):
            self._optimizer, self._learning_rate = TFOptimizers.instantiate_optimizer(self, self.optimizer_tuple)

    def set_training_op(self):

        #def flatten_weights(weights_list):
        #    weights_tensor = []
        #    for w in weights_list:
        #        print(w)
        #        a = tf.reshape(w, [-1, ])
        #        weights_tensor.append(a)
        #
        #    return tf.concat(weights_tensor, axis=0)

        '''
        #########################################
        # Euclidean gradient computed in two steps, through the Jacobian
        #########################################

        #self.probabilities = tf.nn.softmax(self.logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y, tf.int32), logits = self.logits) + self.regularizer
        #middle_man = self.logits # self.natural_loss #

        #n = self.probabilities.get_shape().as_list()[1]
        self.euclidean_gradient = tf.gradients(loss, self.logits)[0] # from [g] to g

        trainable_vars = tf.trainable_variables()
        jacobians = jacobian(self.logits, trainable_vars) # [tf.reduce_sum(i, axis=0) for i in jacobian(self.logits, trainable_vars)]

        #self.nat_grad = [tf.reduce_mean(i, axis=0) for i in self.jacobian]

        # proper way to compute the contraction
        self.euclidean_gradient = [tf.reduce_mean(tf.einsum(get_contraction(i), i, self.euclidean_gradient), axis=0) for i in jacobians]
        # OLD self.nat_grad = [tf.reduce_mean(tf.tensordot(i, self.nat_grad_theta, [[1], [1]]), axis=[0,  1, -1]) for i in self.jacobian]

        self.euclidean_grad_norms = [tf.norm(g) for g in self.euclidean_gradient]

        '''

        total_loss = self.loss
        # add regularizers in case there are any
        if len(self.regularizers)>0:
            total_loss += tf.add_n(self.regularizers, name="regularization")

        # 1st part of minimize: compute_gradient
        self.grads_and_vars = self._optimizer.compute_gradients(total_loss)

        # clip gradients
        clipped_grads_and_vars = self._clip_gradients(self.grads_and_vars, self._grad_clipping_tuple)

        # compute norms in case they need to be logged
        self.gradient_norms = [tf.norm(g) + NUMTOL for (g, v) in clipped_grads_and_vars]
        self.weight_norms = [tf.norm(v) + NUMTOL for (g, v) in clipped_grads_and_vars]
        # check that gradients are finite
        grads = [tf.check_numerics(g, "grads is not finite") for (g, v) in clipped_grads_and_vars]
        variables = [tf.check_numerics(v, "grads is not finite") for (g, v) in clipped_grads_and_vars]
        self.gradient_weight_global_norms = [tf.global_norm(grads), tf.global_norm(variables)]

        # 2nd part of minimize: apply_gradient
        optimizer_step = self._optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

        update_ops = tf.group(*self.update_ops)
        self.training_op = tf.group(update_ops, optimizer_step)


    def set_check_ops(self):
        self._check_ops = 1

        # TODO argo2 This is not working anymore with the new session
        #with self.sess.graph.as_default():
        self._numerics_ops = tf.add_check_numerics_ops()

    def release(self):
        super().release()
        self.sess.close()
        tf.reset_default_graph()

    def set_summaries(self):
        """This function sets summaries and summaryFileWriters, it needs to be invoked before
        training to keep track of the summaries.
        (cannot be invoked in create_and_init_network because the FileWriter will corrupt data in the logfolder
        at each initialization)
        """

        # wb_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # ac_regularizers = tf.get_collection(AC_REGULARIZATION)
        #
        # reg_nodes = wb_regularizers + ac_regularizers

        # for rn in reg_nodes:
        #     tf.summary.scalar(rn.name, rn,
        #                     collections=[tf.GraphKeys.SUMMARIES, 'regularization_summaries'])

        # for each key I get the collection of summary nodes
        # I set up a filewriter for each summary node
        self.summary_nodes = {sk: tf.get_collection(sk) for sk in self.summary_keys}

        for sk in self.summary_keys:
            self.summary_writers[sk] = [tf.compat.v1.summary.FileWriter(self._tensorboard_dir+sn.name)
                                        for sn in self.summary_nodes[sk]]


    def create_hooks(self, config):

        hooks=[]

        # get general arguments for the models hook
        self._time_reference_str = config["time_reference"]
        self._check_time_reference(self._time_reference_str)
        self._plot_offset = config.get("plot_offset", 0)
        self._default_model_hooks_kwargs = {"time_reference" : self._time_reference_str}

        self._plot_model_hooks_kwargs = {"time_reference" : self._time_reference_str,
                                         "plot_offset": self._plot_offset}

        self._n_steps_stats = self._get_steps(config["stats_period"], self._time_reference_str)

        # stop hook
        tot_steps = int(self._opts['epochs']+1)*self.n_batches_per_epoch
        hooks.append(tf.train.StopAtStepHook(last_step=tot_steps))

        # general info hook (no average on validation but only on train loop)
        hooks.append(self._create_general_info_hook(config))

        # regularizers hook (no average on validation but only on train loop)
        hooks.append(self._create_regularizers_hook(config))

        # checkpoint hooks
        self._save_model = config["save_model"]
        if self._save_model:
            max_to_keep = config.get("save_max_to_keep", 5)
            self._init_session_saver(max_to_keep)
            self._checkpoint_basename = "model.ckpt"
            save_steps = self._get_steps(config["save_model_period"], self._time_reference_str)

            hooks.append(CheckpointSaverHook(self._checkpoint_dir,
                                             save_steps = save_steps,
                                             saver = self._saver,
                                             checkpoint_basename = self._checkpoint_basename,
                                             pb_output_nodes = self._pb_output_nodes,
                                             save_pb_at_end = config.get("save_pb", 0)
                                             ))

        # summary hook
        if config["save_summaries"]:
            save_steps_summaries = self._get_steps(config["save_summaries_period"], self._time_reference_str)

            self.set_summaries()

            summary_hooks = [tf.train.SummarySaverHook(save_steps=save_steps_summaries,
                                                       output_dir=self._tensorboard_dir+sn.name,
                                                       summary_op=sn,
                                                       summary_writer=fw)
                             for sk in self.summary_keys for sn,fw in zip(self.summary_nodes[sk], self.summary_writers[sk])]

            hooks += summary_hooks

        # images input hook
        kwargs = config.get("ImagesInputHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}

            hooks.append(ImagesInputHook(model = self,
                                         dirName = self.dirName,
                                         **kwargs)
                         )

        #gradient_hook = self._create_gradient_hook(config)
        #if gradient_hook is not None:
        #    hooks.append(gradient_hook)


        kwargs = config.get("FisherMatrixHook", None)
        if kwargs and isinstance(self._optimizer, NaturalGradientOptimizer):
            kwargs = {**self._default_model_hooks_kwargs,
                      #'datasets_keys' : [TRAIN_LOOP],
                      **kwargs}
            hooks.append(FisherMatrixHook(model = self,
                                          dirName = self.dirName,
                                          **kwargs
                                         )
                        )

        return hooks

    def _create_gradient_hook(self, config):

        # gradienthook
        tensors_to_average = [
            [[self.gradient_weight_global_norms[0]],
             self.gradient_norms
             ],
            [[self.gradient_weight_global_norms[1]],
             self.weight_norms
             ],
        ]

        layer_names = np.array(list(range(len(self.gradient_norms))))
        layer_names = np.floor(layer_names / 2) + 1
        layer_names = ["L" + str(int(l)) for l in layer_names]

        tensors_to_average_names = [
            [["gradient_global_norms"],
             layer_names
             ],
            [["weight_global_norms"],
             layer_names
             ],
        ]

        tensors_to_average_plots = [
            [{"fileName": "gradient_global_norms", "logscale-y": 1, "compose-label": 0},
             {"fileName": "gradient_norms", "logscale-y": 1, "compose-label": 0}
             ],
            [{"fileName": "weight_global_norms", "logscale-y": 1, "compose-label": 0},
             {"fileName": "weight_norms", "logscale-y": 1, "compose-label": 0}
             ],
        ]

        kwargs = config.get("GradientsHook", None)
        if kwargs:
            gradient_period = config["GradientsHook"]["period"]
            gradient_steps = self._get_steps(gradient_period, self._time_reference_str)
            hook = LoggingMeanTensorsHook(model=self,
                                          fileName="gradient",
                                          dirName=self.dirName,
                                          tensors_to_average=tensors_to_average,
                                          tensors_to_average_names=tensors_to_average_names,
                                          tensors_to_average_plots=tensors_to_average_plots,
                                          average_steps=gradient_steps,
                                          tensorboard_dir=self._tensorboard_dir,
                                          trigger_summaries=config["save_summaries"],
                                          # trigger_plot=True,
                                          print_to_screen=False,
                                          plot_offset=self._plot_offset, # config.get("plot_start_epoch", 1),
                                          train_loop_key=TRAIN_LOOP,
                                          datasets_keys=[],
                                          time_reference=self._time_reference_str
                                         )

            return hook
        else:
            return None

    # create custom regularizers id
    # passing the network equal to None support the possibility to use this function in presence
    # of multiple networks, used in gan and vae, not in hm
    def create_custom_regularizers_id(self, network=None):

        if network is None:
            regularizers = self._opts["regularizers"]
        else:
            regularizers = self._opts["regularizers"][network]
        
        ids = ""
        if "custom" in regularizers.keys():
            #import pdb;pdb.set_trace()
            for regularizer_tuple in regularizers["custom"]:

                regularizer_name =  regularizer_tuple[0]

                try:
                    base_path = '.'.join(__name__.split('.')[:-3])
                    regularizer_module = load_module("Regularizers", base_path=base_path)
                    id =  regularizer_module.create_id(regularizer_tuple)
                except Exception as e:
                    # try to load from argo
                    try:
                        id =  Regularizers.create_id(regularizer_tuple)
                    except Exception as e:
                        raise Exception("regularizer %s not found" % regularizer_name) from e

                if ids == "":
                    ids = id
                else:
                    ids = ids + "_" + id
                
        return ids

    
    def _create_regularizers_hook(self, config):

        wb_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # see keras_utils.py: activity_and_contractive_regularizers
        ac_regularizers = tf.get_collection(AC_REGULARIZATION)
        custom_regularizers = tf.get_collection(CUSTOM_REGULARIZATION)

        if wb_regularizers:
            wb_regularizers_names = [r.name for r in wb_regularizers]
        else:
            wb_regularizers = [tf.zeros([1])]
            wb_regularizers_names = ["none"]
        wb_regularizers_fileNames = {"fileName" : "wb_regularizers"}

        if ac_regularizers:
            ac_regularizers_names = [r.name for r in ac_regularizers]
        else:
            ac_regularizers = [tf.zeros([1])]
            ac_regularizers_names = ["none"]
        ac_regularizers_fileNames = {"fileName" : "ac_regularizers"}

        if custom_regularizers:
            custom_regularizers_names = [r.name for r in custom_regularizers]
        else:
            custom_regularizers = [tf.zeros([1])]
            custom_regularizers_names = ["none"]
        custom_regularizers_fileNames = {"fileName": "custom_regularizers"}

        # logging hooks
        tensors_to_average = [[wb_regularizers], [ac_regularizers, custom_regularizers]]
        tensors_to_average_names = [[wb_regularizers_names], [ac_regularizers_names, custom_regularizers_names]]
        tensors_to_average_plots = [[wb_regularizers_fileNames],
                                    [ac_regularizers_fileNames, custom_regularizers_fileNames]]


        hook = LoggingMeanTensorsHook(model=self,
                                      fileName="regularizers",
                                      dirName=self.dirName,
                                      tensors_to_average=tensors_to_average,
                                      tensors_to_average_names=tensors_to_average_names,
                                      tensors_to_average_plots=tensors_to_average_plots,
                                      average_steps=self._n_steps_stats,
                                      tensorboard_dir=self._tensorboard_dir,
                                      trigger_summaries=config["save_summaries"],
                                      print_to_screen=False,
                                      # trigger_plot = True,
                                      plot_offset=self._plot_offset,
                                      train_loop_key=TRAIN_LOOP,
                                      datasets_keys=[],
                                      time_reference=self._time_reference_str
                                     )
        return hook


    def _create_general_info_hook(self, config):
        # logging hooks
        tensors_to_average = [
            [[self._learning_rate]]
        ]
        tensors_to_average_names = [
            [["learning_rate"]],
        ]
        tensors_to_average_plots = [
            [{"fileName": "learning_rate"}]
        ]

        hook = LoggingMeanTensorsHook(model=self,
                                      fileName="info",
                                      dirName=self.dirName,
                                      tensors_to_average=tensors_to_average,
                                      tensors_to_average_names=tensors_to_average_names,
                                      tensors_to_average_plots=tensors_to_average_plots,
                                      average_steps=self._n_steps_stats,
                                      tensorboard_dir=self._tensorboard_dir,
                                      trigger_summaries=config["save_summaries"],
                                      print_to_screen=False,
                                      # trigger_plot = True,
                                      plot_offset = self._plot_offset,
                                      train_loop_key=TRAIN_LOOP,
                                      datasets_keys=[]
                                     )
        return hook


    # why passing opt?
    def create_session(self, opts, config, monitorSession=True):

        # save to set the right behavior in self.get_raw_session()
        self.monitorSession = monitorSession
        
        # set some important options
        if self._gpu == -1:
            sess_config = tf.ConfigProto(device_count = {'GPU': 0},
                                         allow_soft_placement=True)
        else:
            #config = tf.ConfigProto(log_device_placement=True)
            sess_config = tf.ConfigProto(allow_soft_placement=True)

        sess_config.gpu_options.allow_growth = True

        # self.sess = tf.Session(config=config)
        # self.sess = tf.InteractiveSession()

        # not needed anymore, moved in hooks...
        # self.set_summaries()

        if self._check_ops:
            self.set_check_ops()

        self.hooks = self.create_hooks(config)

        #TODO-ARGO2 if we would use a SingularMonitoredSession, it is possible to directly pass it to a saver for custom user saving..
        #TODO-ARGO2 How to handle this with the more stable Monitored Session? Maybe a TFTrainableDeepLearningModel
        #TODO-ARGO2 by the way it is possible to define a custom Monitored session
        #TODO-ARGO2 (to handle only hooks without fancy session stuffs http://davideng.me/2017/10/11/designing-a-custom-monitored-training-session.html
        

        if monitorSession:
            # MonitoredSession
            # this will restore all the variables from the latest checkpoint if it exists
            self._fix_checkpoint_abs_to_rel(self._checkpoint_dir) # need to ensure checkpoint has relative path saved

            chiefsess_creator = tf.train.ChiefSessionCreator(config=sess_config, checkpoint_dir=self._checkpoint_dir)

            if self._restore_chkptfile is not None:
                self._network.init_saver()
            # this is restoring variables 
            self.sess = tf.train.MonitoredSession(session_creator=chiefsess_creator, hooks=self.hooks)

            # Restore from some checkpoint
            if self._restore_chkptfile is not None:
                raw_sess = self.get_raw_session()
                if raw_sess.run(self.global_step) == 0:
                    self._network.restore(raw_sess, self._restore_chkptfile)
        else:
            self.sess = tf.Session(config=sess_config)
            #all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
            #self.sess.run(tf.variables_initializer(all_variables))
        
        if self._save_model:
            self._save_graph()

        # SingularMonitoredSession
        # self.sess = tf.train.SingularMonitoredSession(checkpoint_dir=self._checkpoint_dir,
        #                                              hooks=self.hooks, config=sess_config)


        #I do not want to trigger hooks for this!!
        self.datasets_handles = self.get_raw_session().run(self.datasets_handles_nodes)


    # to get the raw session in MonitoredSession see
    # https://github.com/tensorflow/tensorflow/issues/8425
    # https://github.com/tensorflow/tensorflow/issues/11971
    def get_raw_session(self):
        if self.sess is None:
            raise Exception("The session is None")

        if self.monitorSession:
            return self.sess._tf_sess()
        else:
            # suppose regular Session()
            return self.sess


    def train(self):

        # epoch = 0
        # # start timer for TF
        # start_tf = timeit.default_timer()

        # INITIALIZATION OPERATIONS HERE
        # Also: I always perform a global_step evaluation to eventually "wake up" the
        # stop condition of StopAtStepHook if needed (otherwise it would do an extra step over its limit each time)
        # (possible bug in tf?)
        # initializer = self.dataset_initializers["train"]
        # args = [initializer] if initializer is not None else []
        # args += [self.global_step]
        # self.get_raw_session().run(args)

        for hook in self.hooks:
            before_training = getattr(hook, 'before_training', None)
            if before_training is not None:
                before_training(self.get_raw_session())

        #i = 0

        # count of the nan / inf
        k = 0
        # MAX nan / inf
        MAX_K = 100

        print("Graph size: " + str(self.graph_size))

        # loops over the batches
        while not self.sess.should_stop():
            # import pdb;pdb.set_trace()
            try:
                #import pdb; pdb.set_trace()
                # sess = self.get_raw_session()
                # sess.run(self._network.get_all_variables()[6])
                # g, g_norm, x, y, loss, grads_and_vars, _, global_epoch = self.sess.run([self.grads, self.grads_norm, self.x, self.y, self.loss, self.grads_and_vars , self.training_op, self.global_epoch],


                # loss must be evaluated and fetched to raise InvalidArgumentError if nan, see https://github.com/tensorflow/tensorflow/issues/11098
                _, _, global_epoch = self.sess.run([self.training_op, self.loss, self.global_epoch],
                                                   feed_dict = {self.ds_handle : self.datasets_handles[TRAIN_LOOP],
                                                                self.is_training : True})

                #pdb.set_trace()

                # self.sess.run([self.z],feed_dict = {self.ds_handle : self.datasets_handles[TRAIN_LOOP],self.is_training : True})[0].shape
                # pdb.set_trace()

                #self.invFisher, self.prob,
                #print(i, loss)
                #if i==180:
                #    pdb.set_trace()
                #i = i + 1

            except tf.errors.InvalidArgumentError:

                raise Exception("an error has occurred during training, check stack trace UP HERE")

                k = k+1
                print("an error has occurred during training, this happened " + str(k) + " times over " + str(MAX_K))
                if k == MAX_K:
                    raise Exception("an error has occurred during training, check stack trace UP HERE")



                # here the are two possibilities we could try:
                # 1) ignore the current batch and proceed to the next, in case the error is a numerical error
                # 2) add noise the the weights, see self.random_update_op (so far it didn't help)
                # TODO not sure if I need to evaluate global_epoch or not in this case

                #_, global_epoch = self.sess.run([self.random_update_op, self.global_epoch], feed_dict = {self.ds_handle :  self.datasets_handles[TRAIN_LOOP]})


            # if global_epoch != epoch:
            #     # end timer for TF
            #     stop_tf = timeit.default_timer()
            #     time = stop_tf - start_tf
            #
            #     epoch = global_epoch
            #
            #     print("global epoch: ", global_epoch, " time:",  time)
            #
            #     # start timer for TF
            #     start_tf = timeit.default_timer()

    def _init_session_saver(self, max_to_keep, variables=None):
        """ A saver with all the variables for the session is instantiated and set in self._saver, with variables,
        by default variables is None, all variables in the graph will be saved.
        It is probably a good idea since the whole session must be later be restored by the ChiefSession
        """
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        #variables = tf.trainable_variables()
        self._saver = tf.train.Saver(variables, max_to_keep=max_to_keep, save_relative_paths=True)

    def _save_graph(self):
        writer = tf.summary.FileWriter(logdir=self._checkpoint_dir,
                                       # graph=self.sess.graph,
                                       graph=tf.get_default_graph(),
                                       filename_suffix="-graph"
                                       )
        writer.flush()

    def _assemble_checkpoint_name(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "model.ckpt")
        return path

    def _latest_checkpoint(self, checkpoint_dir):
        with open(checkpoint_dir + 'checkpoint') as fs:
            potentiallyabsolutepath = fs.readline().split()[1]

        potentiallyabsolutepath = os.path.basename(potentiallyabsolutepath.strip('"'))
        path = checkpoint_dir + os.path.basename(potentiallyabsolutepath)
        return path

    def _fix_checkpoint_abs_to_rel(self, checkpoint_dir):
        checkpointfilename = checkpoint_dir + 'checkpoint'
        exists = os.path.isfile(checkpointfilename)
        if exists:
            with open(checkpointfilename) as fs:
                lines = fs.readlines()

            fs = open(checkpointfilename, 'w')
            for line in lines:
                which_model, potentiallyabsolutepath = line.split()
                potentiallyabsolutepath = os.path.basename(potentiallyabsolutepath.strip('"'))
                rel_path = '\"'+os.path.basename(potentiallyabsolutepath)+'\"'
                fs.write(" ".join([which_model, rel_path]) + "\n")

            fs.close()

    def checkpoint_name(self, global_step):
        if global_step:
            path = self._assemble_checkpoint_name(self._checkpoint_dir)
            path += "-"+str(global_step)
        else:
            path = self._latest_checkpoint(self._checkpoint_dir)

        if not path:
            raise Exception("could not find saved checkpoints in %s"%self._checkpoint_dir)

        return path

    def save(self, global_step=None):
        if self._saver is None:
            raise Exception("saver must be initialized before attempt to save")
        else:
            session = self.get_raw_session()
            path = self._assemble_checkpoint_name()
            self._saver.save(session, path, global_step=global_step)

    def restore(self, global_step=None):
        """Restore the model variables.

        Args:
            global_step (type): the step from which to restore. By default it is None
                    and the latest checkpoint in self.checkpoint_dir will be restored
        """

        path = ""
        session = self.get_raw_session()

        if self._saver is None:
            raise Exception("saver must be initialized before attempt to restore")
        else:
            path = self.checkpoint_name(global_step)
            self._saver.restore(session, path)

    @property
    def graph_size(self):
        return len([n.name for n in self.sess.graph.as_graph_def().node])

    def _check_time_reference(self, time_ref):
        time_choices = [EPOCHS, STEPS]
        if not time_ref in time_choices:
            raise ValueError("time_reference in the frequency tuple can only be in %s" % time_choices)

    def _get_steps(self, n, time_reference):
        # try:
        #     n, time_ref = frequency
        # except:
        #     raise Exception("cannot unpack tuple %s, expected a couple (n, time_ref),"
        #                     "where time_ref can either be `epochs` or `steps`" % frequency)

        self._check_time_reference(time_reference)
        n = float(n)

        if time_reference == EPOCHS:
            n = n * self.n_batches_per_epoch

        return int(n)
