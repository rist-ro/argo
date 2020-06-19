import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

import seaborn as sns
import tensorflow as tf
#import tensorflow_transform as tft

# from tensorflow import logging as tf_logging
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

from argo.core.utils.argo_utils import create_concat_opts

import numpy as np
import glob

#from tensorflow.train import SessionRunArgs

from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook

#import os
#import timeit
from argo.core.utils.argo_utils import create_list_colors, tf_cov_times_n_points

from datasets.Dataset import TRAIN, VALIDATION #, TEST, checks_dataset_keys_not_loop,
from datasets.Dataset import linestyle_dataset


class PCALatentVariablesHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 tensors,
                 tensors_names,
                 period,
                 time_reference,
                 dirName,
                 datasets_keys=[TRAIN, VALIDATION], 
                 create_heatmap=0,
                 plot_offset=0):

        dirName = dirName + '/pca_latent'        
        # fileName should be set before calling super()
        #self._fileName = "PCA latent -" + self.network_name + "-num_images-" + str(n_images)
        #self._fileHeader = '# frechet_inception_distance, network = ' + self.network_name + ', num images = ' + str(n_images) + '\n'
        #self._fileHeader += '# period \t fid_train \t fid_validation \t time_sec'

        super().__init__(model,
                         period,
                         time_reference,
                         datasets_keys,
                         dirName=dirName,
                         plot_offset=plot_offset)

        self._handle = model.ds_handle
        self._ds_initializers = model.datasets_initializers
        self._ds_handles_nodes = model.datasets_handles_nodes

        #sself._period = period

        self._tensors = tensors
        self._tensors_names = tensors_names
        self._create_heatmap = create_heatmap == 1

        #images = {ds_key : (index_list, model.dataset.get_elements(index_list, ds_key)) \
        #                    for (ds_key, index_list) in images_indexes.items()}

        #check_dataset_keys_not_loop(list(images.keys()))

        #self._images = images
        #self._datasets_keys = datasets_keys
        #check_dataset_keys_not_loop(datasets_keys)

        self._hook_name = "pca_latent_variables_hook"

        tf_logging.info("Create PCALatentVariablesHook for: " +  ", ".join(datasets_keys))

        # with an abuse of notation, we use this as a dictionary instead of the list of lists of lists
        self._tensors_values = {}
        self._tensors_values["sigma"]  = {}
        
        # in this specific logger this is not used
        #self._files = []
                
    def _begin_once(self):

        self.concat_ops = {}
        self.concat_update_ops = {}
        self.concat_reset_ops = {}
        self.sigma = {}

        for ds_key in self._datasets_keys:

            self.concat_ops[ds_key] = {}
            self.concat_update_ops[ds_key] = {}
            self.concat_reset_ops[ds_key] = {}
            self.sigma[ds_key] = {}

            # every hook needs to have its own accumulator to not have problem of erasing memory that other hooks still needs
            # maybe memory occupation could be improved if really needed, but great troubles for concurrency in parallel execution
            scope = self._hook_name+"/"+ ds_key + "_concat_metric/"

            #with tf.variable_scope(scope) as scope:

            for (tensor, tensor_name)  in zip(self._tensors, self._tensors_names):

                dim = tensor.shape.as_list()[1]

                self.concat_ops[ds_key][tensor_name],\
                    self.concat_update_ops[ds_key][tensor_name],\
                    self.concat_reset_ops[ds_key][tensor_name] = create_concat_opts(scope + tensor_name, tensor)

                # see https://tomaxent.com/2018/01/17/PCA-With-Tensorflow/
                # I need to reshape, since the shape is [0,dim]
                covariance = tf_cov_times_n_points(tf.reshape(self.concat_ops[ds_key][tensor_name],[-1,dim]))

                # see https://stats.stackexchange.com/questions/314046/why-does-andrew-ng-prefer-to-use-svd-and-not-eig-of-covariance-matrix-to-do-pca
                singular_values, u, _ = tf.svd(covariance)
                self.sigma[ds_key][tensor_name] = singular_values

                # singular_values, u, _ = tf.svd(self.concat_ops[ds_key][tensor_name])
                # sigma = tf.diag(singular_values)
                # # first two compoments
                # sigma_reduced = tf.slice(sigma, [0, 0], [dim, 2])
                # #couldn't simply take tf.diag(singular_values[:2])?
                #
                # # X = U S V^T
                # self.pca[ds_key][tensor_name] = tf.matmul(tf.reshape(u,[-1,dim]), sigma_reduced)
                # #these are all the projections of all the points on the 2 principal axis.

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)

    def do_when_triggered(self, run_context, run_values):
        #tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))
        tf_logging.info("trigger for PCALatentVariablesHook")
        #import pdb; pdb.set_trace()

        concat_ops = {}

        for ds_key in self._datasets_keys:
            #images = self._images[ds_key][1]

            session = run_context.session
            dataset_initializer = self._ds_initializers[ds_key]

            #for tensor_name in self._tensors_names:
            session.run([dataset_initializer] + [*self.concat_reset_ops[ds_key].values()])

            while True:
                try:
                    session.run([*self.concat_update_ops[ds_key].values()], feed_dict = {self._handle: self._ds_handles[ds_key] }) # **feed_dict,
                except tf.errors.OutOfRangeError:
                    break

            returns = session.run([*self.sigma[ds_key].values()])
            self._tensors_values["sigma"][ds_key] = dict(zip(self.sigma[ds_key].keys(),returns))

            returns = session.run([*self.concat_ops[ds_key].values()])
            concat_ops[ds_key] = dict(zip(self.concat_ops[ds_key].keys(),returns))

            # unfortunately I cannot compute the real covariance matrix in TF,
            # since I cannot easily know at runtime the number of points.
            # see the comment in utils.tf_cov_time_n_points (Luigi)
            # to solve this issue, I get the number of points from the len of the concat_op node
            for tensor_name in self._tensors_names:
                self._tensors_values["sigma"][ds_key][tensor_name] /= len(concat_ops[ds_key][tensor_name])

        # save to txt file
        for j, tensor_name in enumerate(self._tensors_names):
            for ds_key in self._datasets_keys:
                sigma = self._tensors_values["sigma"][ds_key][tensor_name]
                
                fileName = "pca_" + tensor_name + "_" + str(ds_key) + "_" + self._time_reference_str + "_" + str(self._time_ref).zfill(4)
                
                # save txt file
                data = np.hstack([sigma])
                np.savetxt(self._dirName + '/' + fileName + '.txt', data, fmt='%.3f')

    def plot(self):
        fig = plt.figure(figsize=(10, 6))

        list_colors = create_list_colors(len(self._tensors_names))
        for j, tensor_name in enumerate(self._tensors_names):
            for ds_key in self._datasets_keys:
                sigma = self._tensors_values["sigma"][ds_key][tensor_name]
                eigenvals_idx = np.linspace(0, len(sigma), len(sigma))

                plt.plot(eigenvals_idx, sigma, linestyle_dataset[ds_key], c=list_colors[j % len(list_colors)], label=tensor_name + " " + ds_key)
                
        plt.title(self._plot_title, fontsize=9, loc='center')
        plt.xlim(1, len(sigma)-1)
        plt.ylim(bottom=0)
        #plt.xticks(np.arange(1,len(sigma)+1))
        plt.legend()

        fileName = "pca_" + self._time_reference_str + "_" + str(self._time_ref).zfill(4)

        plt.savefig(self._dirName + '/' + fileName + '.png') # , bbox_inches='tight'
        plt.close()

        if self._create_heatmap:
            self._plot_heatmap()
            
    def _plot_heatmap(self):
        for tensor_name in self._tensors_names:
            for ds_key in self._datasets_keys:
                filenames = sorted(glob.glob(
                    self._dirName + "/pca_" + tensor_name + "_" + str(ds_key) + "_" + self._time_reference_str + "_*.txt"))
                recorded_steps = [t.split("_")[-1][:-4] for t in filenames]

                eigenvals_all_epochs = []
                for filename in filenames:
                    eigenvals = np.loadtxt(filename)
                    eigenvals_all_epochs.append(eigenvals)

                eigenvals_all_epochs = np.reshape(eigenvals_all_epochs, [len(eigenvals_all_epochs), -1]).T
                # normalize the eigenvalues
                eigenvals_all_epochs /= np.max(eigenvals_all_epochs, axis=0)

                ax = sns.heatmap(eigenvals_all_epochs, vmin=0, vmax=1, cmap='Blues')
                plt.xlabel('Time period')
                plt.ylabel('Eigenvalues')
                plt.xticks(np.arange(len(recorded_steps)), recorded_steps)

                heatmap_name = "heatmap_" + tensor_name + "_" + str(ds_key) + "_" + self._time_reference_str + "_" + recorded_steps[-1]
                plt.savefig(self._dirName + '/' + heatmap_name + '.png')
                plt.close()

