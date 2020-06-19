from ..argoLogging import get_logger
tf_logging = get_logger()

import numpy as np
import pdb

from .EveryNEpochsTFModelImagesHook import EveryNEpochsTFModelImagesHook

from ..utils.ImagesSaver import ImagesSaver


class ImagesInputHook(EveryNEpochsTFModelImagesHook):
    """
    Hook showing input images to the datasets
    """
    def __init__(self,
                 model,
                 period,
                 time_reference,
                 how_many,
                 n_images_columns,
                 dirName,                 
                 until = 10, # Luigi: I think 10 is better than 100
                 slice_wise = None,
                 pm_one=True
                 ):

        # fileName should be set before calling super()
        self._fileName = "input"
        dirName = dirName + '/input_images'

        super().__init__(model, period, time_reference, dirName=dirName, pm_one=pm_one)
        self._how_many = how_many
        self._n_images_columns = n_images_columns

        self._last_step_to_log = model._get_steps(until, time_reference)

        #self._has_target = False
        #if hasattr(self._model, 'x_target'):
        #    self._has_target = True

        tf_logging.info("Create ImagesInputHook")

        # add nodes to be computed in before_run
        self._nodes_to_be_computed_by_run["raw_inputs"] = self._model.ds_raw_x #raw_x
        self._nodes_to_be_computed_by_run["aug_inputs"] = self._model.ds_aug_x #x
        #if self._has_target:
        self._nodes_to_be_computed_by_run["perturbed_inputs"] = self._model.ds_perturb_x #x_target

        self._slice_wise = slice_wise

    def do_when_triggered(self, run_context, run_values):

        # I plot input images until last period arrives
        if self._global_step <= self._last_step_to_log:

            tf_logging.info("trigger for ImagesInputHook")
            images = run_values.results["raw_inputs"][:self._how_many]
            images_aug = run_values.results["aug_inputs"][:self._how_many]

            #l = 2
            #if self._has_target:
            images_perturbed = run_values.results["perturbed_inputs"][:self._how_many]
            l = 3

            if self._slice_wise == None:
                rows = int(np.ceil(len(images) / self._n_images_columns))
                panel = [[] for x in range(l * rows)]

                c = 0
                for i in range(0, l * rows, l):
                    for j in range(self._n_images_columns):
                        panel[i].append(images[c])
                        panel[i + 1].append(images_aug[c])
                        #if self._has_target:
                        panel[i + 2].append(images_perturbed[c])
                        if c == len(images) - 1:
                            break
                        else:
                            c = c + 1
            else:
                rows = int(np.ceil(len(images) / self._n_images_columns)) * images.shape[3]
                panel = [[] for x in range(rows * 3)]

                for k in range(images.shape[3]):

                    selected_images = images[:, :, :, k]
                    reshaped_images =  selected_images[:, :, :, None]
                    selected_images_aug = images_aug[:, :, :, k]
                    reshaped_images_aug = selected_images_aug[:, :, :, None]
                    selected_images_perturbed = images_perturbed[:, :, :, k]
                    reshaped_images_perturbed = selected_images_perturbed[:, :, :, None]
                    c = 0
                    for i in range(0, l * rows, l * images.shape[3]):
                        i = i + k * l
                        for j in range(self._n_images_columns):
                            panel[i].append(reshaped_images[c])
                            panel[i + 1].append(reshaped_images_aug[c])
                            #if self._has_target:
                            panel[i + 2].append(reshaped_images_perturbed[c])
                            if c == len(images) - 1:
                                break
                            else:
                                c = c + 1

            self.images_saver.save_images(panel,
                                     fileName = self._fileName + "_" + self._time_reference_str + "_" + str(self._time_ref).zfill(4),
                                     title="1) raw 2) augmented 3) perturbed " + self._fileName,
                                     fontsize=9)
