from ..utils.ImagesSaver import ImagesSaver
from .EveryNEpochsTFModelHook import EveryNEpochsTFModelHook


class EveryNEpochsTFModelImagesHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName=None,
                 pm_one=True,
                 extra_feed_dict={}):
        
        dataset_keys = []
        super().__init__(model,
                         period,
                         time_reference,
                         dataset_keys,
                         dirName=dirName,
        extra_feed_dict=extra_feed_dict)

        # TODO make this check at runtime on the node passed tf.shape(node)
        # TODO this will allow to remove x_shape from dataset (confusing and not needed,
        # TODO indeed for certain datasets the shape can vary and also can vary from train to test, due to different cropping)
        self._image_shape = model.dataset.x_shape # x_shape is general for all datasets

        self._pm_one = pm_one
        self.images_saver = ImagesSaver(self._dirName, pm=self._pm_one)
        if len(self._image_shape) != 3:
            raise Exception("image format not correct: dataset x_shape = `%s`" % str(self._image_shape))

        # since EveryNEpochsTFModelImagesHook is not writing any file, set the list to []
        #elf._files = []

    # this hook does not create txt files
    #def _create_or_open_file(self):
    #    self._file = None

    # this hook does not create txt files
    #def _reset_file(self, session):
    #    pass


    # no plotting for EveryNEpochsImagesHook
    def plot(self):
        pass
