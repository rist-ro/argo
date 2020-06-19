# from tensorflow import logging as tf_logging
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

from argo.core.hooks.AbstractLinearInterpolationHook import AbstractLinearInterpolationHook

from argo.core.utils.ImagesSaver import ImagesSaver

#import os
#import timeit
#from .utils.argo_utils import create_reset_metric

class VAELinearInterpolationHook(AbstractLinearInterpolationHook):

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for VAELinearInterpolationHook")

        self.load_images_once(run_context.session)
                
        for (ds_key) in self._images:

            couples_of_images = self._images[ds_key][1]
            reconstructed_interpolations_means, reconstructed_interpolations_zs = self._model._compute_linear_interpolation(couples_of_images, self._n_images, run_context.session)
            reconstructed_interpolations_fisher_rao = self._model._compute_fisher_rao_interpolation(couples_of_images, self._n_images, run_context.session)

            images_saver = ImagesSaver(self._dirName)

            rows = len(couples_of_images)
            panel = [[] for x in range(rows*3)]
            
            c = 0
            for i in range(0, 3*rows, 3):
                for j in range(self._n_images+2): # include first and last image
                    panel[i].append(reconstructed_interpolations_means[c])
                    panel[i+1].append(reconstructed_interpolations_zs[c])
                    panel[i+2].append(reconstructed_interpolations_fisher_rao[c])
                    #panel[i+2].append(reconstructed_images[c])
                    #if c == len(images)-1:
                    #    break
                    #else:
                    c = c + 1

            # "[1st] interpolation in mu before sampling [2nd] iterpolation in z after sampling"
            images_saver.save_images(panel,
                                     fileName = "interpolation_" + str(ds_key) + "_" + self._time_reference_str + "_" + str(self._time_ref).zfill(4),
                                     title = "interpolations 1) means 2) zs 3) fisher-rao on q(z|x) \n" + self._plot_title,
                                     fontsize=9)
