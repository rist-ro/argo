import pdb

# from tensorflow import logging as tf_logging
from argo.core.argoLogging import get_logger
tf_logging = get_logger()

from .EveryNEpochsTFModelImagesHook import EveryNEpochsTFModelImagesHook

from datasets.Dataset import check_dataset_keys_not_loop


class AbstractImagesReconstructHook(EveryNEpochsTFModelImagesHook):
    """
    Abstract Hook to reconstruct images from an AutoEncoder
    """
    def __init__(self,
                 model,
                 period,
                 time_reference,
                 images_indexes,
                 n_images_columns,
                 dirName,
                 slice_wise=None,
                 pm_one=True,
                 conditional=False
                 ):

        # fileName should be set before calling super()
        self._fileName = "reconstructed" 
        dirName = dirName + '/reconstructed_images'

        super().__init__(model, period, time_reference, dirName, pm_one)

        self._images_indexes = images_indexes
        self._n_images_columns = n_images_columns
        self._images = None
        self._masks = None
        self._labels = None
        self._slice_wise = slice_wise
        self._conditional = conditional

        tf_logging.info("Create ImagesReconstructHook for: \n"+ \
                        "\n".join([ds_key+ ": " + ", ".join(map(str, idxs)) \
                                    for ds_key, idxs in self._images_indexes.items()]))
    
        
    def load_images(self, session):

        if self._images==None:
            check_dataset_keys_not_loop(list(self._images_indexes.keys()))

            images = {ds_key : (index_list, self._model.dataset.get_elements(self._model.x, self._ds_handle, self._ds_handles[ds_key], self._ds_initializers[ds_key], session, index_list)) \
                      for (ds_key, index_list) in self._images_indexes.items()}
            # I set something like the following structure, e.g.
            # images = {TRAIN : ([0,100,200,300,400,500], train_images),
            #           VALIDATION : ([0,100,200,300], validation_images),
            #          },
            
            self._images = images


    def load_masks(self, session):

        if self._masks==None and self._model.mask is not None:
            check_dataset_keys_not_loop(list(self._images_indexes.keys()))

            masks = {ds_key : (index_list, self._model.dataset.get_elements(self._model.mask, self._ds_handle, self._ds_handles[ds_key], self._ds_initializers[ds_key], session, index_list)) \
                      for (ds_key, index_list) in self._images_indexes.items()}
            # I set something like the following structure, e.g.
            # images = {TRAIN : ([0,100,200,300,400,500], train_images),
            #           VALIDATION : ([0,100,200,300], validation_images),
            #          },
            
            self._masks = masks

    def load_labels(self, session):

        if self._conditional and self._labels is None:
            check_dataset_keys_not_loop(list(self._images_indexes.keys()))

            labels = {ds_key: (index_list, self._model.dataset.get_elements(self._model.y, self._ds_handle,
                                                                            self._ds_handles[ds_key],
                                                                            self._ds_initializers[ds_key], session,
                                                                            index_list)) \
                      for (ds_key, index_list) in self._images_indexes.items()}

            self._labels = labels
