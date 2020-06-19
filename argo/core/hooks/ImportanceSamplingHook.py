import tensorflow as tf

# from tensorflow import logging as tf_logging
from ..argoLogging import get_logger

tf_logging = get_logger()

import numpy as np
from scipy import special as spl

from ..utils.argo_utils import create_reset_metric
# from datasets.Dataset import check_dataset_keys_not_loop

from .EveryNEpochsTFModelHook import EveryNEpochsTFModelHook

from datasets.Dataset import TRAIN, VALIDATION


class ImportanceSamplingHook(EveryNEpochsTFModelHook):
    def __init__(self,
                 model,
                 datasets_keys,
                 period,
                 time_reference,
                 tensors_to_average,
                 n_samples,
                 batch_size,
                 repetitions,
                 dirName,
                 dataset_keys=[TRAIN, VALIDATION],
                 plot_offset=0,
                 extra_feed_dict={}
                 ):

        dirName = dirName + '/importance_sampling'

        super().__init__(model,
                         period,
                         time_reference,
                         dataset_keys,
                         dirName=dirName,
                         plot_offset=plot_offset,
                         extra_feed_dict=extra_feed_dict)

        self._n_samples = n_samples
        self._batch_size = batch_size
        self._repetitions = repetitions
        self._label_n_samples = "_".join([str(i) for i in self._n_samples])

        self._tensors_to_average = tensors_to_average

        tf_logging.info("Create ImportanceSampling for %s samples %d repetitions " % (
            " ".join([str(i) for i in self._n_samples]), self._repetitions))

        # nodes to be computed and saved
        names = ["is-n" + str(n_samples) + "-r" + str(rep) for n_samples in self._n_samples for rep in
                 range(self._repetitions)]

        fileName = "importance_sampling-n" + "_".join([str(i) for i in n_samples])
        self._tensors_names = [[names]]
        self._tensors_plots = [[{
            'fileName': fileName,
            'logscale-y': 0}]]
        self._tensors_values = {}
        self._fileName = "importance_sampling" + "_" + self._label_n_samples

    """
    Hook for importance sampling estimation
    """

    def _begin_once(self):

        self._mean_values = {}
        self._mean_update_ops = {}
        self._mean_reset_ops = {}

        for dataset_str in self._datasets_keys:
            self._mean_values[dataset_str], self._mean_update_ops[dataset_str], self._mean_reset_ops[dataset_str] = zip(
                *[create_reset_metric(tf.metrics.mean, scope=dataset_str + "_mean_reset_metric/" + tnsr.name + "/" + self._label_n_samples,
                                      values=tnsr) for tnsr in self._tensors_to_average])

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for ImportanceSampling")

        for dataset_str in self._datasets_keys:
            np_values = []
            for n in self._n_samples:

                for i in range(self._repetitions):
                    mean_value = 0

                    imp_sampling = None
                    to_be_done = n
                    for j in range(int(np.ceil(n / self._batch_size))):

                        k = min(to_be_done, self._batch_size)
                        to_be_done = to_be_done - k

                        init_ops = (self._ds_initializers[dataset_str])
                        run_context.session.run(init_ops)
                        imp_sampling_ds = None
                        while True:
                            try:
                                imp_sampling_batch = run_context.session.run(self._tensors_to_average,
                                                                             feed_dict={
                                                                                 self._model.n_z_samples: k,
                                                                                 self._ds_handle:
                                                                                                          self._ds_handles[
                                                                                                              dataset_str],

                                                                                 **self._extra_feed_dict}
                                                                             )
                                if imp_sampling_ds is None:
                                    imp_sampling_ds = imp_sampling_batch
                                else:
                                    imp_sampling_ds = np.concatenate([imp_sampling_ds, imp_sampling_batch], axis=1)
                            except tf.errors.OutOfRangeError:
                                break
                        # now I have a complete pass over the dataset, for the jth chunk
                        if imp_sampling is None:
                            imp_sampling = imp_sampling_ds
                        else:
                            imp_sampling = np.concatenate([imp_sampling, imp_sampling_ds], axis=0)

                    imp_sampling_group_chunks = spl.logsumexp(imp_sampling, axis=0)
                    imp_sampling_group_chunks -= np.log(n)
                    mean_value = np.mean(imp_sampling_group_chunks)

                    np_values.append(mean_value)

            self._tensors_values[dataset_str] = [[np_values]]

        self.log_to_file_and_screen()
