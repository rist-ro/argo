# from tensorflow import logging as tf_logging
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

import numpy as np

from .EveryNEpochsTFModelImagesHook import EveryNEpochsTFModelImagesHook

from ..utils.ImagesSaver import ImagesSaver


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
                 slice_wise=None,
                 pm_one=True
                 ):

        # fileName should be set before calling super()
        self._fileName = "generated"
        self._dirName = dirName + '/generated_images'
        super().__init__(model, period, time_reference, dirName=self._dirName, pm_one=pm_one)

        self._n_images_columns = n_images_columns
        self._n_gen_samples = n_gen_samples
        self._slice_wise = slice_wise

        tf_logging.info("Create ImagesGenerateHook for %d samples" % self._n_gen_samples)


    def do_when_triggered(self, run_context, run_values):
        #tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))
        tf_logging.info("trigger for ImagesGenerateHook")

        images = self._model.generate(batch_size = self._n_gen_samples, sess=run_context.session)

        if self._slice_wise == None:
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
        else:
            rows = int(np.ceil(len(images) / self._n_images_columns)) * images.shape[3]
            panel = [[] for x in range(rows)]

            for k in range(images.shape[3]):
                selected_images = images[:, :, :, k]
                reshaped_images = selected_images[:, :, :, None]

                c = 0
                for i in range(0, rows, images.shape[3]):
                    i = i + k
                    for j in range(self._n_images_columns):
                        panel[i].append(reshaped_images[c])
                        if c == len(images) - 1:
                            break
                        else:
                            c = c + 1

        self.images_saver.save_images(panel,
                                 fileName = self._fileName + "_" + self._time_reference_str + "_" + str(self._time_ref).zfill(4),
                                 title = "generated image " + self._fileName,
                                 fontsize=9)
