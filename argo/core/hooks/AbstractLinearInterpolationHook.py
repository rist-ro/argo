from argo.core.argoLogging import get_logger
tf_logging = get_logger()

from .EveryNEpochsTFModelImagesHook import EveryNEpochsTFModelImagesHook

from datasets.Dataset import check_dataset_keys_not_loop

class AbstractLinearInterpolationHook(EveryNEpochsTFModelImagesHook):
    """
    Abstract Hook for linear interpolation of images from an AutoEncoder
    """
    def __init__(self,
                 model,
                 period,
                 time_reference,
                 images_indexes,
                 n_images,
                 dirName,
                 pm_one=True
                 ):

        self._dirName = dirName + '/linear_interpolations'

        super().__init__(model, period, time_reference, dirName=self._dirName, pm_one=pm_one)

        self._images_indexes = images_indexes
        self._images = None
        self._n_images = n_images

        tf_logging.info("Create LinearInterpolationHook for: \n"+ \
                        "\n".join([ds_key+": "+", ".join(map(str, idxs)) \
                                    for ds_key, idxs in self._images_indexes.items()]))

    def do_when_triggered(self, run_context, run_values):
        pass

    def load_images_once(self, session):        
        if self._images==None:
            check_dataset_keys_not_loop(list(self._images_indexes.keys()))

            images = {ds_key : (couple_indices_list, [(img1, img2) for img1, img2 in zip(
            self._model.dataset.get_elements(self._model.x, self._ds_handle, self._ds_handles[ds_key], self._ds_initializers[ds_key], session, [i[0] for i in couple_indices_list]),
            self._model.dataset.get_elements(self._model.x, self._ds_handle, self._ds_handles[ds_key], self._ds_initializers[ds_key], session, [i[1] for i in couple_indices_list]))]) \
                for (ds_key, couple_indices_list) in self._images_indexes.items()}

            # I set something like the following structure, e.g.
            # images = {TRAIN : ([(0,50),(100,230),(200,790),(300,600),(400,1000),(500,10)], list_of_couples_train_images,
            #           VALIDATION : ([(0,50),(100,230),(200,790),(300,600),(400,1000),(500,10)], list_of_couples_validation_images,
            #          },        
            
            self._images = images


    # no plotting for Linear Interpolations
    def plot(self):
        pass
