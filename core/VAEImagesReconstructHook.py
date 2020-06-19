# from tensorflow import logging as tf_logging
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

import pdb

import numpy as np

from argo.core.hooks.AbstractImagesReconstructHook import AbstractImagesReconstructHook

from argo.core.utils.ImagesSaver import ImagesSaver

#import os
#import timeit
#from .utils.argo_utils import create_reset_metric


class VAEImagesReconstructHook(AbstractImagesReconstructHook):

    def do_when_triggered(self, run_context, run_values):
        #tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))

        tf_logging.info("trigger for ImagesReconstructHook")

        self.load_images(run_context.session)
        self.load_masks(run_context.session)
        self.load_labels(run_context.session)

        for ds_key in self._images:
            y = self._labels[ds_key][1] if self._conditional else None

            images = self._images[ds_key][1]
            
            zs, (means, _) = self._model.encode(images, sess=run_context.session, y=y)

            if self._model.mask is None:
                reconstructed_images_no_sampling = self._model.decode(means, sess=run_context.session, y=y)
                reconstructed_images = self._model.decode(zs, sess=run_context.session, y=y)
            else:
                masks = self._masks[ds_key][1]
                reconstructed_images_no_sampling = self._model.decode(means, sess=run_context.session, mask=masks, y=y)
                reconstructed_images = self._model.decode(zs, sess=run_context.session, mask=masks, y=y)
                
            images_saver = ImagesSaver(self._dirName)

            if self._slice_wise == None:
                rows = int(np.ceil(len(images) / self._n_images_columns))
                panel = [[] for x in range(rows * 3)]

                c = 0
                for i in range(0, 3 * rows, 3):
                    for j in range(self._n_images_columns):
                        panel[i].append(images[c])
                        panel[i + 1].append(reconstructed_images_no_sampling[c])
                        panel[i + 2].append(reconstructed_images[c])
                        if c == len(images) - 1:
                            break
                        else:
                            c = c + 1
            else:
                rows = int(np.ceil(len(images) / self._n_images_columns)) * images.shape[3]
                panel = [[] for x in range(rows * 3)]

                for k in range(images.shape[3]):
                    selected_images = images[:, :, :, k]
                    reshaped_images = selected_images[:, :, :, None]
                    selected_reconstructed_images_no_sampling = reconstructed_images_no_sampling[:, :, :, k]
                    reshaped_reconstructed_images_no_sampling = selected_reconstructed_images_no_sampling[:, :, :, None]
                    selected_reconstructed_images = reconstructed_images[:, :, :, k]
                    reshaped_reconstructed_images = selected_reconstructed_images[:, :, :, None]

                    c = 0
                    for i in range(0, 3 * rows, 3 * images.shape[3]):
                        i = i + k * 3
                        for j in range(self._n_images_columns):
                            panel[i].append(reshaped_images[c])
                            panel[i + 1].append(reshaped_reconstructed_images_no_sampling[c])
                            panel[i + 2].append(reshaped_reconstructed_images[c])
                            if c == len(images) - 1:
                                break
                            else:
                                c = c + 1

            # "[1st] original image [2nd] recostructed  mean [3rd] reconstr z"
            images_saver.save_images(panel,
                                     fileName = "reconstructed" + "_" + str(ds_key) + "_" + self._time_reference_str + "_" + str(self._time_ref).zfill(4),
                                     title = self._plot_title,
                                     fontsize=9)
