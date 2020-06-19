# from tensorflow import logging as tf_logging
from argo.core.hooks.EveryNEpochsTFModelImagesHook import EveryNEpochsTFModelImagesHook
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

import numpy as np

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
                 dirName,
                 pm_one=True,
                 ):

        self._dirName = dirName + '/generated_images'
        super().__init__(model, period, time_reference, dirName=self._dirName,
                                               pm_one=pm_one)
        
        self._n_images_columns = n_images_columns
        self._n_gen_samples = n_gen_samples
        
        tf_logging.info("Create ImagesGenerateHook for %d samples" % self._n_gen_samples)


    def do_when_triggered(self, run_context, run_values):
        #tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))
        tf_logging.info("trigger for ImagesGenerateHook")

        images_mean, images_sample  = self._model.generate(batch_size = self._n_gen_samples, sess=run_context.session)

        rows = int(np.ceil(len(images_sample)/self._n_images_columns))
        
        panel = [[] for x in range(rows)]
        c = 0
        for i in range(rows):
            for j in range(self._n_images_columns):
                panel[i].append(images_sample[c])
                if c == len(images_sample) - 1:
                    break
                else:
                    c = c + 1

        self.images_saver.save_images(panel,
                                 fileName = "generated_" + self._time_ref_shortstr + "_" + str(self._time_ref).zfill(4),
                                 title = self._plot_title,
                                 fontsize=9)
