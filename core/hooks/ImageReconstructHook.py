from argo.core.argoLogging import get_logger

tf_logging = get_logger()

import numpy as np

from argo.core.hooks.AbstractImagesReconstructHook import AbstractImagesReconstructHook

from argo.core.utils.ImagesSaver import ImagesSaver

class ImagesReconstructHook(AbstractImagesReconstructHook):

    def do_when_triggered(self, run_context, run_values):
        # tf_logging.info("trigger for ImagesGeneratorHook s" +  str(global_step) + " s/e" + str(global_step/global_epoch)+ " e" + str(global_epoch))
        tf_logging.info("trigger for ImagesReconstructHook")

        self.load_images(run_context.session)

        for ds_key in self._images:
            images = self._images[ds_key][1]
            zs, means = self._model.encode(images, run_context.session)
            reconstructed_images_m_means, reconstructed_images_m_sample = self._model.decode(means, run_context.session)
            reconstructed_images_z_means, reconstructed_images_z_sample = self._model.decode(zs, run_context.session)

            rows = int(np.ceil(len(images) / self._n_images_columns))
            panel = [[] for x in range(rows * 5)]

            c = 0
            for i in range(0, 5 * rows, 5):
                for j in range(self._n_images_columns):
                    panel[i].append(images[c])
                    panel[i + 1].append(reconstructed_images_m_means[c])
                    panel[i + 2].append(reconstructed_images_m_sample[c])
                    panel[i + 3].append(reconstructed_images_z_means[c])
                    panel[i + 4].append(reconstructed_images_z_sample[c])
                    if c == len(images) - 1:
                        break
                    else:
                        c = c + 1

            # "[1st] original image [2nd] recostructed  mean [3rd] reconstr z"
            self.images_saver.save_images(panel,
                                     fileName="reconstruction_" + str(ds_key) + "_" + self._time_ref_shortstr + "_" + str(
                                         self._time_ref).zfill(4),
                                     title=self._plot_title,
                                     fontsize=9)
