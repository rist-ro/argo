import tensorflow as tf

from ..argoLogging import get_logger

tf_logging = get_logger()

from ..utils.argo_utils import create_reset_metric, compose_name

from datasets.Dataset import TRAIN_LOOP, VALIDATION

from .ArgoHook import ArgoHook

import os


def evaluate_means_over_dataset(session, handle, dataset_initializer, dataset_handle,
                                metrics_values, metrics_update_ops, metrics_reset_ops, feed_dict = {}, max_iterations = -1):

    if type(session).__name__ != 'Session':
        raise Exception("I need a raw session to evaluate metric over dataset.")

    session.run(dataset_initializer)
    session.run(metrics_reset_ops)

    # pdb.set_trace()
    iteration = 0
    while max_iterations ==-1 or iteration < max_iterations:
        try:
            session.run(metrics_update_ops, feed_dict = {**feed_dict,
                                                         handle: dataset_handle})
            iteration +=1

        except tf.errors.OutOfRangeError:
            break

    return session.run(metrics_values)

def evaluate_over_dataset(session, handle, dataset_initializer, dataset_handle,
                          metrics_values, feed_dict = {}, max_iterations = -1):

    if type(session).__name__ != 'Session':
        raise Exception("I need a raw session to evaluate over dataset.")

    session.run(dataset_initializer)

    # pdb.set_trace()
    iteration = 0
    while max_iterations ==-1 or iteration < max_iterations:
        try:
            session.run(metrics_update_ops, feed_dict = {**feed_dict,
                                                         handle: dataset_handle})
            iteration +=1

        except tf.errors.OutOfRangeError:
            break

    return session.run(metrics_values)

'''
def evaluate_means_over_dataset_sample(session, fid, metrics_values, metrics_update_ops, metrics_reset_ops, feed_dict = {}, max_iterations = -1):

    if type(session).__name__ != 'Session':
        raise Exception("I need a raw session to evaluate metric over dataset.")
    # pdb.set_trace()

    session.run(metrics_reset_ops)


    iteration = 0

    while max_iterations ==-1 or iteration < max_iterations:
        session.run(metrics_update_ops, feed_dict = {fid.samples:fid.random_samples})
        iteration +=1

    return session.run(metrics_values)
'''

class LoggingMeanTensorsHook(ArgoHook):
    """
    Needed to Average certain variables every N steps
    """

    def __init__(self,
                 model,
                 fileName,
                 dirName,
                 tensors_to_average,
                 tensors_to_average_names,
                 tensors_to_average_plots,
                 average_steps,
                 tensorboard_dir,
                 trigger_summaries,
                 # trigger_plot = True,
                 print_to_screen=True,
                 plot_offset=0,
                 train_loop_key=TRAIN_LOOP,
                 datasets_keys=[VALIDATION],
                 time_reference="epochs"
                 ):

        """LoggingMeanTensorsHook.

        """
        super(LoggingMeanTensorsHook, self).__init__(model,
                                                     average_steps,
                                                     time_reference,
                                                     datasets_keys,
                                                     plot_offset=plot_offset,
                                                     tensorboard_dir=tensorboard_dir,
                                                     trigger_summaries=trigger_summaries)
        print("LoggingMeanTensorsHook")

        assert (tensors_to_average)
        assert (tensors_to_average_names)
        assert (tensors_to_average_plots)

        self._default_metric = tf.metrics.mean
        
        self._fileName = fileName

        self._dirName = dirName
        if dirName:
            os.makedirs(dirName, exist_ok = True)
            
        # check if the second panel is empty, as it happens in case the cost function
        # is returning an empty list for "self.loss_nodes_to_log"
        # (not super elegant done in this way, however good enough for the moment - Luigi)
        if len(tensors_to_average) > 1 and not tensors_to_average[1]:
            del tensors_to_average[1]
            del tensors_to_average_names[1]
            del tensors_to_average_plots[1]

        self._tensors = tensors_to_average

        # nodes computed and saved
        self._tensors_names = tensors_to_average_names
        self._tensors_plots = tensors_to_average_plots
        self._tensors_values = []

        self._print_to_screen = print_to_screen

        self._train_loop_key = train_loop_key

        # parent constructor already called, so refer to self
        self._no_train_loop_datasets_keys = self._datasets_keys
        self._datasets_keys = [self._train_loop_key] + self._datasets_keys

        for vertical_panels in self._tensors_names:
            for tensor_names_panel in vertical_panels:
                tf_logging.info("Create LoggingMeanTensorsHook for: " + ", ".join(tensor_names_panel))

        self._did_begin_already = False

    def begin(self):
        super(LoggingMeanTensorsHook, self).begin()

        if not self._did_begin_already:
            self._did_begin_already = True

            self._mean_values = {}
            self._mean_update_ops = {}
            self._mean_reset_ops = {}

            for dataset_str in self._datasets_keys:

                mean_values_panel = []
                mean_update_ops_panel = []
                mean_reset_ops_panel = []

                for vertical_panels in self._tensors:

                    mean_values = []
                    mean_update_ops = []
                    mean_reset_ops = []

                    for tensors_panel in vertical_panels:
                        mean_v, mean_u_ops, mean_r_ops = \
                            zip(*[create_reset_metric(self._default_metric,
                                                      scope=dataset_str + "_mean_reset_metric/" + tnsr.name,
                                                      values=tnsr)
                                  for tnsr in tensors_panel])

                        mean_values.append(mean_v)
                        mean_update_ops.append(mean_u_ops)
                        mean_reset_ops.append(mean_r_ops)

                    mean_values_panel.append(mean_values)
                    mean_update_ops_panel.append(mean_update_ops)
                    mean_reset_ops_panel.append(mean_reset_ops)

                self._mean_values[dataset_str] = mean_values_panel
                self._mean_update_ops[dataset_str] = mean_update_ops_panel
                self._mean_reset_ops[dataset_str] = mean_reset_ops_panel

                # These fors could be merged, I'm just not sure it will help the readability
                for (mean_values_vertical_panels, tensors_names_vertical_panels) in zip(self._mean_values[dataset_str],
                                                                                        self._tensors_names):
                    for (mean_values_panel, tensor_names_panel) in zip(mean_values_vertical_panels,
                                                                       tensors_names_vertical_panels):
                        for mn, mn_name in zip(mean_values_panel, tensor_names_panel):
                            self._register_summary_for_tensor(compose_name(mn_name, dataset_str), mn)

            if self._global_step_tensor is None:
                raise RuntimeError("Global step should be created to use LoggingMeanTensorsHook.")

            self._create_or_open_files()

    def _before_run_args(self):
        args = super(LoggingMeanTensorsHook, self)._before_run_args()

        for (tensors_plots_vertical_panels, mean_update_vertical_panels) in zip(self._tensors_plots,
                                                                                self._mean_update_ops[
                                                                                    self._train_loop_key]):
            for (tensors_plots, mean_update_ops) in zip(tensors_plots_vertical_panels, mean_update_vertical_panels):
                # the name of the op is the name of the entire subplot, which is file of the txt file
                args = {**args,
                        "update_ops_" + tensors_plots["fileName"]: mean_update_ops}

        return args

    def _after_run(self, run_context, run_values):
        pass
    
    def after_run(self, run_context, run_values):
        super(LoggingMeanTensorsHook, self).after_run(run_context, run_values)

        if self._trigged_for_step:

            # update time and reset triggered for step
            self.update_time()

            # mean over train_loop
            self._tensors_values = {}
            # self._mean_values
            self._tensors_values[self._train_loop_key] = run_context.session.run(
                self._mean_values[self._train_loop_key])

            # mean over the other dataset_keys
            for dataset_str in self._no_train_loop_datasets_keys:
                self._tensors_values[dataset_str] = evaluate_means_over_dataset(run_context.session,
                                                                                self._ds_handle,
                                                                                self._ds_initializers[dataset_str],
                                                                                self._ds_handles[dataset_str],
                                                                                self._mean_values[dataset_str],
                                                                                self._mean_update_ops[dataset_str],
                                                                                self._mean_reset_ops[dataset_str])

            self._after_run(run_context, run_values)

            self.log_to_file_and_screen(self._print_to_screen)

            self.reset_means(run_context.session)

            if self._time_ref < 50 or self._time_ref % 10 == 0:
                self.plot()

    def reset_means(self, session):
        for mean_reset_ops_panel, tensors_vertical_panel in zip(self._mean_reset_ops[self._train_loop_key],
                                                                self._tensors_names):

            if len(tensors_vertical_panel) > 0:

                for mean_reset_ops in mean_reset_ops_panel:
                    # reset train means
                    session.run(mean_reset_ops)
