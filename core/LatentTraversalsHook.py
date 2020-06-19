# from tensorflow import logging as tf_logging
from argo.core.hooks.EveryNEpochsTFModelImagesHook import EveryNEpochsTFModelImagesHook
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

import numpy as np

from argo.core.utils.ImagesSaver import ImagesSaver


from datasets.Dataset import check_dataset_keys_not_loop


class LatentTraversalsHook(EveryNEpochsTFModelImagesHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 images_indexes,
                 n_images_columns,
                 radius,
                 step,
                 dirName
                 ):

        self._dirName = dirName + '/latent_traversals'

        super().__init__(model, period, time_reference, dirName=self._dirName)

        self._fileName = "latent_traversal"

        self._images_indexes = images_indexes
        self._n_images_columns = n_images_columns
        self._radius = radius
        self._step = step
        tf_logging.info("Create LatentTraversalsHook for: \n" + \
                        "\n".join([ds_key + ": " + ", ".join(map(str, idxs)) \
                                   for ds_key, idxs in self._images_indexes.items()]))

    def load_images(self, session):
        check_dataset_keys_not_loop(list(self._images_indexes.keys()))

        images = {ds_key: (index_list, self._model.dataset.get_elements(self._model.x, self._ds_handle,
                                                                        self._ds_handles[ds_key],
                                                                        self._ds_initializers[ds_key], session,
                                                                        index_list)) \
                  for (ds_key, index_list) in self._images_indexes.items()}

        self._images = images

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for LatentTraversalsHook")

        self.load_images(run_context.session)

        # create values to be added to latent variables
        offset_range = np.arange(1, self._radius + 1)
        half_offsets = offset_range * self._step
        offsets = np.concatenate([-np.flip(half_offsets), [0], half_offsets])
        z_dim = self._model._gaussian_model_latent.batch_shape.as_list()[1]
        num_traversals = offsets.shape[0]

        offset_matrix = np.zeros([num_traversals * z_dim, z_dim])
        for i in range(z_dim):
            offset_matrix[i * num_traversals: (i + 1) * num_traversals, i] = offsets


        for (ds_key, index_list) in self._images_indexes.items():
            for cnt, img_idx in enumerate(index_list):
                image = self._images[ds_key][1][None, cnt, ...]

                encodings = self._model.encode(image, run_context.session)
                # change the means and decode without sampling
                means = encodings[1]

                tiled_means = np.repeat(means, num_traversals * z_dim, axis=0)

                traversal_means = tiled_means + offset_matrix

                reconstructed_images = self._model.decode(traversal_means, run_context.session)
                all_dims_but_first = list(reconstructed_images.shape[1:])
                reconstructed_images = reconstructed_images.reshape([z_dim, num_traversals] + all_dims_but_first)

                images_saver = ImagesSaver(self._dirName)

                images_saver.save_images(reconstructed_images,
                                         fileName="latent_traversal_" + str(ds_key) + "_" +
                                                    self._time_ref_shortstr + "_" + str(self._time_ref).zfill(4) + "_idx_" +
                                                    str(img_idx),
                                         title=self._fileName,
                                         fontsize=9)
