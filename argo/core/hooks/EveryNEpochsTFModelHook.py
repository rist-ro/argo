import os
from abc import abstractmethod

from .ArgoHook import ArgoHook
from ..argoLogging import get_logger

tf_logging = get_logger()

class EveryNEpochsTFModelHook(ArgoHook):
    """
    Needed to perform certain operations on a TFDeepLearningModel every N epochs
    """

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dataset_keys,
                 dirName=None,
                 log_str=None,
                 tensorboard_dir=None,
                 trigger_summaries=False,
                 extra_feed_dict={},
                 plot_offset=0):
        """
        Args:
            model: the argo model for which to log stuffs
            period: a tuple `(n, time_ref)` to specify when to trigger, time_ref can be "epoch" or "step" used for plotting and txt files
            dirName: directory where to write stuffs

        """

        n_steps = model._get_steps(period, time_reference)
        super().__init__(model, n_steps, time_reference, dataset_keys,
                         plot_offset=plot_offset, tensorboard_dir=tensorboard_dir, trigger_summaries=trigger_summaries,
                         extra_feed_dict=extra_feed_dict)

        self._plot_title = model.dataset.id + " " + model.id


        self._dirName = dirName

        self._did_begin_already = False

        if dirName:
            os.makedirs(dirName, exist_ok=True)

        if log_str:
            tf_logging.info(log_str)

    def begin(self):
        super(EveryNEpochsTFModelHook, self).begin()

        if not self._did_begin_already:
            self._did_begin_already = True

            if self._global_step_tensor is None:
                raise RuntimeError(
                    "Global step should be created to use LoggingMeanTensorsHook.")

            self._begin_once()

        if self._tensors_names is not None and self._tensors_plots is not None and self._tensors_values is not None:
            self._create_or_open_files()
        else:
            print(str(self) + " is not using custom txt files")
            self._files = []
            self._filesExist = []

    @abstractmethod
    def _begin_once(self):
        pass

    def after_run(self, run_context, run_values):
        super(EveryNEpochsTFModelHook, self).after_run(run_context, run_values)

        if self._trigged_for_step:

            # update time and reset triggered for step
            self.update_time()

            self.do_when_triggered(run_context, run_values)

            if (self._time_ref < 50 or self._time_ref % 10 == 0):
                self.plot()

    @abstractmethod
    def do_when_triggered(self, run_context, run_values):
        pass
