import tensorflow as tf
# from tensorflow import logging as tf_logging
from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook
from ..argoLogging import get_logger
tf_logging = get_logger()

from itertools import chain

import numpy as np

#from argo.core.utils.argo_utils import create_reset_metric, compose_name
#from datasets.Dataset import check_dataset_keys_not_loop

#from argo.core.hooks.LoggingMeanTensorsHook import evaluate_means_over_dataset
from .EveryNEpochsTFModelImagesHook import EveryNEpochsTFModelImagesHook

from ..utils.argo_utils import create_list_colors, create_concat_opts #, tf_cov_times_n_points

import sonnet as snt

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

#from argo.core.hooks.ImagesGenerator import ImagesGenerator

from datasets.Dataset import TRAIN_LOOP #, check_dataset_keys_not_loop, linestyle_dataset, short_name_dataset

import pdb

#from sklearn import linear_model

class LatentVarsClassificationHook(EveryNEpochsTFModelHook):
    """
    Hook for lineary separability of latent space
    """

    def __init__(self,
                 model,
                 dirName,
                 tensors,
                 tensors_names,
                 datasets_keys,
                 period,                 
                 time_reference,
                 learning_rate,
                 steps,
                 repetitions,
                 plot_offset=0,
                 extra_feed_dict={}
                 ):

        self._hook_name = "latent_vars_linear_classifier"
        dirName = dirName + '/' + self._hook_name
        # fileName should be set before calling super()
        #self._fileName = "frechet_inception_distance-" + self.network_name + "-num_images-" + str(n_images)
        #self._fileHeader = '# frechet_inception_distance, network = ' + self.network_name + ', num images = ' + str(n_images) + '\n'
        #self._fileHeader += '# period \t fid_train \t fid_validation \t time_sec'

        super().__init__(model,
                         period,
                         time_reference,
                         datasets_keys,
                         dirName=dirName,
                         plot_offset=plot_offset,
                         extra_feed_dict=extra_feed_dict)

        #self._n_samples = n_samples
        #self._batch_size = batch_size
        #self._repetitions = repetitions
        
        # potentially shared with a parent class
        #self._handle = model.ds_handle
        #self._datasets_keys = datasets_keys
        #check_dataset_keys_not_loop(datasets_keys)

        #self._ds_initializers = model.datasets_initializers
        #self._ds_handles_nodes = model.datasets_handles_nodes

        #self._tensors_to_average = tensors_to_average
        #self._tensors_to_average_names = tensors_to_average_names

        self._repetitions = repetitions
        
        self._learning_rate = learning_rate
        self._steps = steps

        labels_avg_cov = list(chain(*[(n + "_runs" + str(self._repetitions) + "_avg", n + "_runs" + str(self._repetitions) + "_cov") for n in tensors_names]))
        self._tensors = [[tensors]]
        self._tensors_names = [[[n + "_r" + str(r) for r in range(self._repetitions) for n in tensors_names]],
        [labels_avg_cov]]
        self._tensors_plots = [[{
            'fileName': "latent_vars_linear_classifier",
            'logscale-y': 0}],
            [{
            'fileName': "latent_vars_linear_classifier_avg",
            'logscale-y': 0,
            'error-bars': 1}]]
        self._tensors_values = {}
        self._fileName = "latent_vars_linear_classifier"

        self._tensor_y_name = "y"
        self._tensor_y = tf.cast(self._model.y, dtype=tf.int64)
        
        tf_logging.info("Create LatentVarsClassification")
                
        
        #self._tensors = [[]]
        #self._tensors_names = [[["latent_vars_classifier"]]]
        #self._tensors_plots = [[{"fileName": "latent_vars_classifier"}]]
        
    # potentially moved in as parent class
    #def after_create_session(self, session, coord):
    #    super().after_create_session(session, coord)
    #    self._ds_handles = session.run(self._ds_handles_nodes)

    '''
    def _begin_once(self):

        self.concat_ops = {}
        self.concat_update_ops = {}
        self.concat_reset_ops = {}
        self.beta = {}

        # every hook needs to have its own accumulator to not have problem of erasing memory that other hooks still needs
        # maybe memory occupation could be improved if really needed, but great troubles for concurrency in parallel execution
        scope = self._hook_name + "/" + "concat_metric/"

        #with tf.variable_scope(scope) as scope:

        # accumulating y
        self.concat_ops[self._tensor_y_name],\
            self.concat_update_ops[self._tensor_y_name],\
            self.concat_reset_ops[self._tensor_y_name] = create_concat_opts(scope + self._tensor_y_name, self._tensor_y)
            
        for (tensor, tensor_name)  in zip(self._tensors, self._tensors_names):

            dim = tensor.shape.as_list()[1]

            self.concat_ops[tensor_name],\
                self.concat_update_ops[tensor_name],\
                self.concat_reset_ops[tensor_name] = create_concat_opts(scope + tensor_name, tensor)

            # see https://tomaxent.com/2018/01/17/PCA-With-Tensorflow/
            # I need to reshape, since the shape is [0,dim]
            #covariance = tf_cov_times_n_points(tf.reshape(self.concat_ops[tensor_name],[-1,dim]))
            x = tf.reshape(self.concat_ops[tensor_name],[-1,dim])
            y = tf.reshape(self.concat_ops[self._tensor_y_name],[-1,dim])
            XtX = tf.matmul(tf.transpose(x), x)
            Xty = tf.reshape(tf.linalg.tensor_diag_part(tf.matmul(tf.transpose(x), y)),[-1,1])

            self.beta[tensor_name] = tf.matmul(tf.linalg.inv(XtX),Xty)
    
        self.placeholder_beta = tf.placeholder("float", [None])
        self.placeholder_y = tf.placeholder("int", [None, 1]) 
        self.accuracy = tf.matmul(self.z,tf.matmul)
    '''

    def _begin_once(self):

        self.tf_metric_update = {}
        self.tf_metric = {}
        self.tf_metric_reset = {}
        self.training_op = {}
        self.loss_train = {}

        self.tf_initializers = {}
        
        n_labels = self._model.dataset.n_labels

        # [Luigi] here I cheat (arguably), since I know that there is only one panel, I avoid 2 extra cycles
        for (tensor, tensor_name) in zip(self._tensors[0][0], self._tensors_names[0][0]):

            dim = tensor.shape.as_list()[1]

            scope = self._hook_name + "/classifier_" + tensor_name
            with tf.name_scope(scope):

                linear_module = snt.Linear(output_size=n_labels)
                logits = linear_module(tensor)
                
                # save initializers to reset weights before retraining
                var_module = linear_module.get_variables()
                self.tf_initializers[tensor_name] = tf.variables_initializer(var_module)

                # check if I need to replicate the labels, due to multiple samples of z for a given x
                replicate_condition = tf.equal(tf.shape(self._tensor_y)[0], tf.shape(tensor)[0])
                y_replicate = tf.cond(replicate_condition,
                                      lambda: self._tensor_y,
                                      lambda: tf.tile(self._tensor_y, [self._model._network.n_z_samples]))                
                
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_replicate)
                loss = tf.reduce_mean(loss)
                self.loss_train[tensor_name] = loss 
                self.tf_metric[tensor_name], self.tf_metric_update[tensor_name] = tf.metrics.accuracy(y_replicate,
                                                                                                      tf.argmax(logits, axis=1),
                                                                                                      name="accuracy_metric")

                # reset metric
                self.tf_metric_reset[tensor_name] = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
                
                # see https://steemit.com/machine-learning/@ronny.rest/avoiding-headaches-with-tf-metrics
                # isolate the variables stored behind the scenes by the metric operation
                running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
                # define initializer to initialize/reset running variables
                running_vars_initializer = tf.variables_initializer(var_list=running_vars)
                optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
                # get local variables
                classifier_vars = [linear_module.w, linear_module.b]
                gradients = optimizer.compute_gradients(loss, var_list=classifier_vars)
                self.training_op[tensor_name] = optimizer.apply_gradients(gradients)
                

    def do_when_triggered(self, run_context, run_values):
        #tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))
        tf_logging.info("trigger for LatentVarsClassificationHook")
        
        session = run_context.session

        self._tensors_values = {}
        accuracies = {}
        
        # train logistic regression
        # [Luigi] here I cheat (arguably), since I know that there is only one panel, I avoid 2 extra cycles

        for r in range(self._repetitions):
            for (tensor, tensor_name) in zip(self._tensors[0][0], self._tensors_names[0][0]):
            
                loss = 0
                n_steps_print = 1000

                # reinitialze the weights of the network in case of multiple repetitions
                session.run(self.tf_initializers[tensor_name])
                    
                for step in range(1,self._steps+1):

                    # average loss, to see if it goes down
                    batch_loss, _ = session.run([self.loss_train[tensor_name], self.training_op[tensor_name]],
                                                feed_dict = {self._ds_handle : self._ds_handles[TRAIN_LOOP]})
                    loss += batch_loss
                
                    if step>0 and step % n_steps_print == 0:
                        print("r:" + str(r) + " step:" + str(step) + ": " + str(loss/n_steps_print))
                        loss = 0
            
                # evaluation of the model
                for ds_key in self._datasets_keys:

                    dataset_initializer = self._ds_initializers[ds_key]
                    session.run([dataset_initializer] + [self.tf_metric_reset[tensor_name]])

                    while True:
                        try:
                            session.run(self.tf_metric_update[tensor_name], feed_dict = {self._ds_handle: self._ds_handles[ds_key]})
                        except tf.errors.OutOfRangeError:
                            break

                    accuracy = session.run(self.tf_metric[tensor_name])
                    #print(ds_key, accuracy)
                    
                    if not ds_key in accuracies.keys():
                        accuracies[ds_key] = []
                    accuracies[ds_key].append(accuracy)

        n_tensors = len(self._tensors[0][0])
        for ds_key in self._datasets_keys:
            # create proper list structure, and computing mean and std
            list_mean_avg = np.mean(np.array(accuracies[ds_key]).reshape([-1,n_tensors]), axis=0).tolist() + np.std(np.array(accuracies[ds_key]).reshape([-1,n_tensors]), axis=0).tolist()
            # move from mean mean mean cov cov cov
            # to mean con mean cov mean cov
            indices_mean_avg = list(chain(*[(n, n + n_tensors) for n in range(n_tensors)]))
            list_mean_avg = np.take(list_mean_avg, indices_mean_avg)
            self._tensors_values[ds_key] = [[accuracies[ds_key]], [list_mean_avg]]
            
        self.log_to_file_and_screen()
                
    def plot(self):
        super().plot()

        '''
        # plot images
        if self._plot_misclassified_images:
            fileName = self._fileName + "-misclass_train_mean-ep"
            image_generator = ImagesGenerator(self._dirName, fileName, self._n_misclassified_images)
            image_generator.set_color(self._color)
            image_generator.image_size = self._shape
            #resized_images = [image.reshape(self._shape) for image in self.misclassified_train_mean ]
            image_generator.save_images(self.misclassified_train_mean, epoch)

            fileName = self._fileName + "-misclass_test_mean-ep"
            image_generator = ImagesGenerator(self._dirName, fileName, self._n_misclassified_images)
            image_generator.set_color(self._color)
            image_generator.image_size = self._shape
            #resized_images = [image.reshape(self._shape) for image in self.misclassified_test_mean ]
            image_generator.save_images(self.misclassified_test_mean, epoch)

            fileName = self._fileName + "-misclass_train_mean_cov-ep"
            image_generator = ImagesGenerator(self._dirName, fileName, self._n_misclassified_images)
            image_generator.set_color(self._color)
            image_generator.image_size = self._shape
            #resized_images = [image.reshape(self._shape) for image in self.misclassified_train_mean_cov ]
            image_generator.save_images(self.misclassified_train_mean_cov, epoch)

            fileName = self._fileName + "-misclass_test_mean_cov-ep"
            image_generator = ImagesGenerator(self._dirName, fileName, self._n_misclassified_images)
            image_generator.set_color(self._color)
            image_generator.image_size = self._shape
            #resized_images = [image.reshape(self._shape) for image in self.misclassified_test_mean_cov ]
            image_generator.save_images(self.misclassified_test_mean_cov, epoch)

            fileName = self._fileName + "-misclass_train_samples-ep"
            image_generator = ImagesGenerator(self._dirName, fileName, self._n_misclassified_images)
            image_generator.set_color(self._color)
            image_generator.image_size = self._shape
            #resized_images = [image.reshape(self._shape) for image in self.misclassified_train_samples ]
            image_generator.save_images(self.misclassified_train_samples, epoch)

            fileName = self._fileName + "-misclass_test_samples-ep"
            image_generator = ImagesGenerator(self._dirName, fileName, self._n_misclassified_images)
            image_generator.set_color(self._color)
            image_generator.image_size = self._shape
            #resized_images = [image.reshape(self._shape) for image in self.misclassified_test_samples ]
            image_generator.save_images(self.misclassified_test_samples, epoch)


        bins = np.linspace(0, 15, 100)

        fig = plt.figure(figsize=(15,10))

        ax1 = fig.add_subplot(321)
        ax1.set_title("Train set means")
        ax1.hist(self.norm_mean_train_class, bins, alpha=0.5, label='train mean class')
        ax1.hist(self.norm_mean_train_misclass, bins, alpha=0.5, label='train mean misclass')
        ax1.legend(loc='upper right')

        ax2 = fig.add_subplot(322)
        ax2.set_title("Test set means")
        ax2.hist(self.norm_mean_test_class, bins, alpha=0.5, label='test mean class')
        ax2.hist(self.norm_mean_test_misclass, bins, alpha=0.5, label='test mean misclass')
        ax2.legend(loc='upper right')

        ax3 = fig.add_subplot(323)
        ax3.set_title("Train set means+covs")
        ax3.hist(self.norm_mean_cov_train_class, bins, alpha=0.5, label='train mean+cov class')
        ax3.hist(self.norm_mean_cov_train_misclass, bins, alpha=0.5, label='train mean+cov misclass')
        ax3.legend(loc='upper right')

        ax4 = fig.add_subplot(324)
        ax4.set_title("Test set means+covs")
        ax4.hist(self.norm_mean_cov_test_class, bins, alpha=0.5, label='test mean+cov class')
        ax4.hist(self.norm_mean_cov_test_misclass, bins, alpha=0.5, label='test mean+cov misclass')
        ax4.legend(loc='upper right')

        ax5 = fig.add_subplot(325)
        ax5.set_title("Train set samples")
        ax5.hist(self.norm_samples_train_class, bins, alpha=0.5, label='train samples class')
        ax5.hist(self.norm_samples_train_misclass, bins, alpha=0.5, label='train sampes misclass')
        ax5.legend(loc='upper right')

        ax6 = fig.add_subplot(326)
        ax6.set_title("Tests set samples")
        ax6.hist(self.norm_samples_test_class, bins, alpha=0.5, label='test samples class')
        ax6.hist(self.norm_samples_test_misclass, bins, alpha=0.5, label='test sampes misclass')
        ax6.legend(loc='upper right')

        plt.savefig(self._dirName + "/" + self._fileName + "-norms_class-ep" + str(self._time_ref).zfill(4) + ".png")
        plt.close()
        '''
