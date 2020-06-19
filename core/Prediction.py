import tensorflow as tf
import tensorflow_probability as tfp

from .preprocessing.preprocess import get_preproc_module
from argo.core.TFDeepLearningModel import TFDeepLearningModel
from argo.core.utils.argo_utils import tf_sample_discrete_from_continuous, \
                tf_add_gaussian_noise_and_clip, get_method_id

from argo.core.network.load_network import instantiate_network
#from argo.core.optimizers.DropOneLogitOptimizer import DropOneLogitOptimizer

from argo.core.hooks.LoggingMeanTensorsHook import LoggingMeanTensorsHook

# from argo.core.hooks.HessianHook import HessianHook
from .hooks.CorrelationHook import CorrelationHook
from .hooks.MCDropoutHook import MCDropoutHook
from .hooks.MCRegressionHook import MCRegressionHook
from .hooks.MCClassificationHook import MCClassificationHook

# from .hooks.MCDropoutHook_alpha import MCDropoutHook_alpha
from .hooks.WeightsHistogramHook import WeightsHistogramHook

from argo.core.CostFunctions import CostFunctions

from datasets.Dataset import TRAIN_LOOP, TRAIN, VALIDATION, TEST
from argo.core.network.KerasNetwork import KerasNetwork
import copy

'''
see load_model, load_class and load_network instead!

def get_prediction_model_class(model_params):
    task = model_params.get("task", None)

    if not task:
        ValueError("Task not found in model_params. task must be set in the parameters of a PredictionModel")

    if task=='classification':
        ModelClass = ClassificationModel
    elif task=='regression':
        ModelClass = RegressionModel
    else:
        raise ValueError("task not recognized can be either: classification or regression. Found `%s` in model_params."%task)

    return ModelClass


def get_prediction_model(model_params, model_dir, check_ops=False, gpu=-1, seed=0):
    ModelClass = get_prediction_model_class(model_params)
    model = ModelClass(model_params, model_dir, check_ops=check_ops, gpu=gpu, seed=seed)
    return model
'''

class PredictionModel(TFDeepLearningModel):

    launchable_name = "FF"

    default_params= {
        **TFDeepLearningModel.default_params,
        #"dtype" : "float32",  # this is completely ignored at the moment
        }

    def __init__(self, opts, dirName, check_ops=False, gpu=-1, seed=0):

        # check if I need the following lines
        #
        # NB need to create the network before the super init because id generation depends on the network
        #self._network = FFNetwork(opts, "ff_network")
        #super().__init__(opts, dirName, check_ops, gpu, seed)
        #
        #self.stochastic = opts["stochastic"]
        #self.stochastic_noise_param = opts["stochastic_noise_param"]
        #self.rescale = opts["rescale"] # rescale the inputs if they are continuous

        ######################################################

        # in the following I pass opts, since the parent constructor has not been called yet
        self._network = instantiate_network(copy.deepcopy(opts), "ff_network")
        self._cost_function = CostFunctions.instantiate_cost_function(opts["cost_function"], module_path= "prediction")

        self.network_output = None
        self._preproc_tuple = opts.get("preprocess", None)

        n_samples_train = int(opts.get("n_samples_train", 1))

        # temporary backcompatibility
        if n_samples_train==1:
            cost_kwargs = opts['cost_function'][1]
            n_samples_train = int(cost_kwargs.get("n_samples", 1))

        self.n_samples_ph = tf.placeholder_with_default(n_samples_train, shape=(), name='n_samples')

        # parent constructor called last
        # notice here I pass opts
        super().__init__(opts, dirName, check_ops, gpu, seed)

        self._pb_output_nodes = ["logits", "ff_network/network/features"]

    def create_id(self):
        
        _id = self.launchable_name

        # add to the ID the information of the cost function
        _id += '-c' + self._cost_function.create_id(self._opts["cost_function"][1])

        #str_type = get_short_dtype(opts["dtype"])
        preproc_tuple = self._opts.get("preprocess", None)
        if preproc_tuple is not None:
            _id += '-p' + get_method_id(preproc_tuple)

        super_id = super().create_id()
        network_id = self._network.create_id()

        _id += super_id + network_id
        if len(self._opts["regularizers"].keys())>0:
            _id += '-cr' + self.create_custom_regularizers_id()
              
        return _id

    def create_network(self):
        # TODO this is not the place for such a specific thing (?)
        # TODO Please drop stuffs in specific places not in standard interfaces. Inheritance avoid `if`.

        '''
        # Now I know this, since the optimizers has been already set
        if isinstance(self._optimizer, DropOneLogitOptimizer):
            self._drop_one_logit = 1
            for i in range(3000):
                print("TODO up this line here. MSE is innocent, does not want drop_one_logit... (?)")
        else:
            self._drop_one_logit = 0
        '''

        # create feed-forward network
        # TODO also here drop stuffs... let us not pollute the interfaces to the networks please. Inherit, and implement specific behaviours.
        self.network_output = self._network(self.x, is_training=self.is_training) #, drop_one_logit=self._drop_one_logit)

        # Keras does not use global collections we need a different way to handle it
        if isinstance(self._network, KerasNetwork):
            reg_losses, kl_losses, update_ops = self._network.get_keras_losses(self.x)
            self.kl_losses += kl_losses
            self.update_ops += update_ops
            self.regularizers += reg_losses

        if isinstance(self.network_output, tfp.distributions.Distribution):
            self.prediction_distr = self.network_output

            #these creates the nodes we need, they do not compute with sampling if possible (as in the Normal case)
            if isinstance(self.prediction_distr, tfp.distributions.TransformedDistribution):
                # mean is not implemented (e.g. flow) so estimate from 10 samples
                self.prediction_mean = tf.reduce_mean(self.prediction_distr.sample(10), axis=0)
            else:
                self.prediction_mean = self.prediction_distr.mean()

            # self.prediction_variance = self.prediction_distr.variance()
            # self.prediction_covariance = self.prediction_distr.covariance()
            self.prediction_sample = tf.squeeze(self.prediction_distr.sample(1), axis=0) # need this squeeze for flow
            self.check_output_shape(self.prediction_mean.get_shape())

        else:

            # set a specific name, so that I can recognize the name in the pb file of the graph
            self.logits = tf.identity(self.network_output, name="logits")
            #pdb.set_trace()
            self.prediction_mean = self.logits
            self.prediction_sample = self.logits
            # self.prediction_covariance = 0
            self.check_output_shape(self.logits.get_shape())

    def create_input_nodes(self, dataset):
        """
        creates input nodes for a feedforward from the dataset

        Sets:
            x, y
        """

        datasets_nodes, handle, ds_initializers, ds_handles = self.create_datasets_with_handles(dataset)

        '''
        #perturbed dataset is not contemplated for the prediction case
        if perturbed_dataset:
            raise Exception("perturbed datasets are not contemplated for the prediction case, use a regular dataset")
        '''
        
        # self.ds_raw_x already set in TFDeepLearningModel.py
        self.raw_y = datasets_nodes[1]
        # set a specific name, so that I can recognize the name in the pb file of the graph
        self.raw_x = tf.identity(self.ds_aug_x, name="inputs")

        self.augment_bool = tf.placeholder_with_default(True, shape=())
        self._x = tf.cond(self.augment_bool,
                         lambda: self._augment_data_nodes(self.raw_x),
                         lambda: self.raw_x
                         )

        self.x = tf.tile(self._x, [self.n_samples_ph, 1, 1, 1])
        self.y_shape = dataset.y_shape
        self.y = self.raw_y
        #self.y = tf.tile(self.raw_y, [self.alpha_samples, 1])

    # this logic is very simple to follow and it seems good AFAIK, please talk to me if you need to modify it.. (Riccardo)
    def _augment_data_nodes(self, dataset_x):

        # I want to do some general transformation of the input. (Riccardo)
        # it is before the noise since this transformation might filter the noise otherwise.. (VAE)
        if self._preproc_tuple:
            self.preproc_module = get_preproc_module(self._preproc_tuple)
            dataset_x = self.preproc_module(dataset_x)

        if self.stochastic:
            if self.binary:
                dataset_x = tf_sample_discrete_from_continuous(dataset_x)
            else:
                # TODO here I suppose the input are in -1.,1.
                dataset_x, noise_data = tf_add_gaussian_noise_and_clip(dataset_x,
                                                                       self.stochastic_noise_param,
                                                                       clip_bool=self._clip_after_noise)

        # TODO NEVER RESCALE remove it
        # we need to shrink data to make sure that we don't have 0 and 1 for continuous
        # data, otherwise we may have "division by 0" for the likelihood of the logit-normal
        # also, this may be usefu for binary data, to avoid gradients equal to zero in the first
        # layer
        # #TODO what to do with rescale, should we rescale also on raw_x? What happens if I am an autoencoder and I want to reconstruct target?
        # if not self.rescale==0.0:
        #     # TODO add a check that the domain is in [-1,1]
        #     dataset_x = tf_rescale(dataset_x, self.rescale)

        return dataset_x

    def create_hooks(self, config):
        hooks = super().create_hooks(config)

        # logging hooks

        log_tensors_to_average = self.loss_nodes_to_log

        log_tensors_to_average_names = self.loss_nodes_to_log_names

        log_tensors_to_average_plots = self.loss_nodes_to_log_filenames

        hooks.append(LoggingMeanTensorsHook(model = self,
                                            fileName = "log",
                                            dirName = self.dirName,
                                            tensors_to_average = log_tensors_to_average,
                                            tensors_to_average_names = log_tensors_to_average_names,
                                            tensors_to_average_plots = log_tensors_to_average_plots,
                                            time_reference=self._time_reference_str,
                                            average_steps = self._n_steps_stats,
                                            tensorboard_dir = self._tensorboard_dir,
                                            trigger_summaries = config["save_summaries"],
                                            #trigger_plot = True,
                                            print_to_screen=True,
                                            plot_offset = self._plot_offset,
                                            train_loop_key = TRAIN_LOOP,
                                            # if you want to remove some dataset from here, make support to specify from conf on which datasets to log, if in doubt ask me please. Riccardo
                                            datasets_keys = [TRAIN, VALIDATION, TEST]
                            )
                     )

        kwargs = config.get("CorrelationHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      'datasets_keys' : [TRAIN, VALIDATION],
                      **kwargs}

            hooks.append(CorrelationHook(model = self,
                                         dirName = self.dirName,
                                         **kwargs
                                         )
                        )

        kwargs = config.get("MCDropoutHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      'datasets_keys': [VALIDATION, TEST],
                      **kwargs}

            hooks.append(MCDropoutHook(model=self,
                                       dirName=self.dirName,
                                       **kwargs))

        kwargs = config.get("MCRegressionHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      'datasets_keys': [VALIDATION, TEST],
                      **kwargs}

            hooks.append(MCRegressionHook(model=self,
                                                   dirName=self.dirName,
                                                   **kwargs))

        kwargs = config.get("MCClassificationHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      'datasets_keys': [VALIDATION, TEST],
                      **kwargs}

            hooks.append(MCClassificationHook(model=self,
                                                   dirName=self.dirName,
                                                   **kwargs))

        # kwargs = config.get("MCDropoutHook_alpha", None)
        # if kwargs:
        #     kwargs = {**self._default_model_hooks_kwargs,
        #               'datasets_keys' : [TEST,VALIDATION],
        #               **kwargs}
        #
        #     hooks.append(MCDropoutHook_alpha(model=self,
        #                                dirName=self.dirName,
        #                                **kwargs))


        kwargs = config.get("WeightsHistogramHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}

            hooks.append(WeightsHistogramHook(model=self,
                                              dirName=self.dirName,
                                              **kwargs))

        '''
        kwargs = config.get("HessianHook", None)
        if kwargs:
            kwargs = {**self._default_model_hooks_kwargs,
                      **kwargs}
            hooks.append(HessianHook(model = self,
                                     dirName = self.dirName,
                                     #tensors = [self.z,
                                     #           self._gaussian_model_latent_mean],
                                     #tensors_names = ['z',
                                     #                 'mu'],
                                     datasets_keys = [TRAIN, VALIDATION], # don't change the order (Luigi)
                                     **kwargs
                                    )
                         )
        '''
            
        return hooks

    def create_loss(self):
        self.loss, self.loss_per_sample, self.loss_nodes_to_log, self.loss_nodes_to_log_names, self.loss_nodes_to_log_filenames = self._cost_function(self)#, drop_one_logit=self._drop_one_logit)



    def predict(self, X, sess=None):
        sess = sess or self.get_raw_session()
        return sess.run([self.prediction_sample], feed_dict={self.x: X})


class ClassificationModel(PredictionModel):

        #self.loss = tf.reduce_mean(
        #                    tf.nn.sparse_softmax_cross_entropy_with_logits(
        #                        labels = tf.cast(self.y, tf.int32),
        #                        logits = self.logits))

        #accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=1),
        #                                           tf.cast(self.y, dtype = tf.int64)),
        #                                  dtype = tf.float32))

        #self.loss_nodes_to_log = [[1-accuracy]]
        #self.loss_nodes_to_log_names = [["error"]]
        #self.loss_nodes_to_log_filenames = [{"fileName" : "error", "logscale-y" : 1}]

    def check_output_shape(self, shape):
        #TODO-ARGO2 why mnist labels are not given in one_hot? and how do I know how many classes I should have here?
        
        #if self._drop_one_logit:
        #    if shape.as_list()[1] != (self.dataset.n_labels-1):
        #        raise Exception("shape `%s` is different from labels shape `%s`,\
#please check network architecture will produce correct output before the softmax")

        #else:
        if shape.as_list()[1] != self.dataset.n_labels:
            raise Exception("shape `%s` is different from labels shape `%s`,\
please check network architecture will produce correct output before the softmax")
        

class RegressionModel(PredictionModel):

    def check_output_shape(self, shape):

        if shape.as_list()[1:] != list(self.dataset.y_shape):
            raise Exception("shape `%s` is different from regression target shape `%s`,\
                    please check network architecture will produce correct output")
