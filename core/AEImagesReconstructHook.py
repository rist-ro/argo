# from tensorflow import logging as tf_logging
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

import numpy as np

from argo.core.hooks.AbstractImagesReconstructHook import AbstractImagesReconstructHook

from argo.core.utils.ImagesSaver import ImagesSaver

#import os
#import timeit
#from .utils.argo_utils import create_reset_metric


class AEImagesReconstructHook(AbstractImagesReconstructHook):

    def do_when_triggered(self, run_context, run_values):
        #tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))
        tf_logging.info("trigger for ImagesReconstructHook")
        time_ref = self._time_ref
        time_ref_str = self._time_ref_shortstr

        self.load_images(run_context.session)

        for ds_key in self._images:
            images = self._images[ds_key][1]

            hs = self._model.encode(images, sess=run_context.session)
            reconstructed_images = self._model.decode(hs, sess=run_context.session)

            images_saver = ImagesSaver(self._dirName)

            rows = int(np.ceil(len(images)/self._n_images_columns))
            panel = [[] for x in range(rows*2)]

            c = 0
            for i in range(0,2*rows,2):
                for j in range(self._n_images_columns):
                    panel[i].append(images[c])
                    panel[i+1].append(reconstructed_images[c])
                    if c == len(images)-1:
                        break
                    else:
                        c = c + 1

            # "[1st] original image [2nd] recostructed  mean [3rd] reconstr z"
            images_saver.save_images(panel,
                                     fileName = "reconstructed" + "_" + str(ds_key) + "_" + time_ref_str+ "_" + str(time_ref).zfill(4),
                                     title = self._fileName,
                                     fontsize=9)
