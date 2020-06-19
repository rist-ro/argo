import tensorflow as tf
# from tensorflow import logging as tf_logging
from .argoLogging import get_logger
tf_logging = get_logger()

import pdb
import os
import numpy as np

from tensorflow.train import SessionRunArgs

from .Hooks import EveryNEpochsTFModelImagesHook

from .ImagesSaver import ImagesSaver


class ImagesGenerateHook(EveryNEpochsTFModelImagesHook):
    """
    Hook Generating images by sampling from the latent space
    """
    
    def __init__(self,
                 model,
                 period,
                 time_reference,
                 n_gen_samples,
                 #n_images_rows,
                 n_images_columns,
                 dirName
                 ):

        self._dirName = dirName + '/generated_images'
        super().__init__(model, period, time_reference, dirName=self._dirName)
        
        self._n_images_columns = n_images_columns
        self._n_gen_samples = n_gen_samples
        
        tf_logging.info("Create ImagesGenerateHook for %d samples" % self._n_gen_samples)


    def do_when_triggered(self, global_step, time_ref, run_context, run_values, time_ref_str="ep"):
        #tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))
        tf_logging.info("trigger for ImagesGenerateHook")

        pdb.set_trace()
        images = self._model.generate(batch_size = self._n_gen_samples, sess=run_context.session)

        images_saver = ImagesSaver(self._dirName)

        rows = int(np.ceil(len(images)/self._n_images_columns))
        
        panel = [[] for x in range(rows)]
        c = 0
        for i in range(rows):
            for j in range(self._n_images_columns):
                panel[i].append(images[c])
                if c == len(images) - 1:
                    break
                else:
                    c = c + 1

        images_saver.save_images(panel,
                                 fileName = "generated_" + time_ref_str + "_" + str(time_ref).zfill(4),
                                 title = self._fileName,
                                 fontsize=9)
