import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

# from tensorflow import logging as tf_logging
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

from argo.core.utils.argo_utils import create_concat_opts

import pdb
import os
import numpy as np

#from tensorflow.train import SessionRunArgs

from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook

#import os
#import timeit
#from .utils.argo_utils import create_reset_metric

from datasets.Dataset import TRAIN, VALIDATION #, TEST, check_dataset_keys_not_loop


class TwoDimPCALatentVariablesHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 tensors,
                 tensors_names,
                 period,
                 time_reference,
                 dirName,
                 datasets_keys = [TRAIN, VALIDATION]
                 ):

        self._dirName = dirName + '/pca_latent'

        super().__init__(model, period, time_reference, dataset_keys=datasets_keys, dirName=self._dirName)

        #self._ds_handle = model.ds_handle
        #self._ds_initializers = model.datasets_initializers
        #self._ds_handles_nodes = model.datasets_handles_nodes

        self._tensors = tensors
        self._tensors_names = tensors_names

        #images = {ds_key : (index_list, model.dataset.get_elements(index_list, ds_key)) \
        #                    for (ds_key, index_list) in images_indexes.items()}

        #check_dataset_keys_not_loop(list(images.keys()))

        #self._images = images
        self._hook_name = "two_dim_pca_latent_variables_hook"
        tf_logging.info("Create TwoDimPCALatentVariablesHook for: " +  ", ".join(self._datasets_keys))

        # in this specific logger this is not used
        self._files = []

    def _begin_once(self):

        self.concat_ops = {}
        self.concat_update_ops = {}
        self.concat_reset_ops = {}
        self.pca = {}

        for ds_key in self._datasets_keys:

            self.concat_ops[ds_key] = {}
            self.concat_update_ops[ds_key] = {}
            self.concat_reset_ops[ds_key] = {}
            self.pca[ds_key] = {}

            # every hook needs to have its own accumulator to not have problem of erasing memory that other hooks still needs
            # maybe memory occupation could be improved if really needed, but great troubles for concurrency in parallel execution
            scope = self._hook_name + "/" + ds_key + "_concat_metric/"

            #with tf.variable_scope(scope) as scope:

            for (tensor, tensor_name)  in zip(self._tensors, self._tensors_names):

                dim = tensor.shape.as_list()[1]

                self.concat_ops[ds_key][tensor_name], self.concat_update_ops[ds_key][tensor_name], self.concat_reset_ops[ds_key][tensor_name] = create_concat_opts(scope + tensor_name, tensor)

                # see https://tomaxent.com/2018/01/17/PCA-With-Tensorflow/
                singular_values, u, _ = tf.svd(self.concat_ops[ds_key][tensor_name])
                sigma = tf.diag(singular_values)
                # first two compoments
                sigma_reduced = tf.slice(sigma, [0, 0], [dim, 2])
                #couldn't simply take tf.diag(singular_values[:2])?

                #TODO are you sure? PCA should use right vectors, usually called V, not U. If convention is X = U S V^T
                # the reshape is needed (why, which is original output shape of u?)
                self.pca[ds_key][tensor_name] = tf.matmul(tf.reshape(u,[-1,dim]), sigma_reduced)
                #queste sono le proiezioni di tutti i punti sui primi 2 principal axis.

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)


    def do_when_triggered(self, run_context, run_values):
        #tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))
        tf_logging.info("trigger for TwoDimPCALatentVariablesHook")

        for ds_key in self._datasets_keys:
            #images = self._images[ds_key][1]

            session = run_context.session

            dataset_initializer = self._ds_initializers[ds_key]

            #for tensor_name in self._tensors_names:
            session.run([dataset_initializer] + [*self.concat_reset_ops[ds_key].values()])

            while True:
                try:
                    session.run([*self.concat_update_ops[ds_key].values()], feed_dict = {self._ds_handle: self._ds_handles[ds_key] }) # **feed_dict,
                except tf.errors.OutOfRangeError:
                    break

            returns = session.run([*self.pca[ds_key].values()])
            tensors_pca = dict(zip(self.pca[ds_key].keys(),returns))

            for tensor_name in self._tensors_names:

                pca = tensors_pca[tensor_name]

                plt.figure(figsize=(10, 10))

                # I need to take into account the number of samples z,
                # and replicate z in a little bit tricky way

                #labels = self._model.dataset.get_labels(ds_key)
                
                labels = self._model.dataset.get_elements(self._model.y, self._ds_handle, self._ds_handles[ds_key], dataset_initializer, session)

                batch_size = self._model.batch_size["eval"]

                #TODO comment: what is samples? use t.shape for tensors (not len(t)) is much more understandable.
                # import pdb;pdb.set_trace()
                samples = int(len(pca)/len(labels)) #self._model.samples

                repeated_labels = [0] * (len(labels)*samples)
                for i in range(0,len(labels),batch_size):
                    repeated_labels[i*samples:i*samples+batch_size*samples] = np.tile(labels[i:i+batch_size], samples)

                plt.scatter(pca[:, 0], pca[:, 1], c=repeated_labels, cmap='gist_rainbow', s=7)

                plt.title(self._plot_title, fontsize=9, loc='center')

                fileName = "pca2d_" + tensor_name + "_" + str(ds_key) + "_" + self._time_reference_str + "_" + str(self._time_ref).zfill(4)

                plt.savefig(self._dirName + '/' + fileName + '.png') # , bbox_inches='tight'
                plt.close()


                # TODO this needs to be replaced

                # save txt file
                data = np.hstack([pca,np.array(repeated_labels).reshape(-1,1)])
                np.savetxt(self._dirName + '/' + fileName + '.txt', data, fmt='%.3f %.3f %i', newline='\r\n')

    def plot(self):
        pass
