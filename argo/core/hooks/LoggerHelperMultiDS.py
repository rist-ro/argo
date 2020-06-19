import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from ..utils.argo_utils import create_reset_metric
from .LoggingMeanTensorsHook import evaluate_means_over_dataset
import itertools

class LoggerHelperMultiDS:

    def __init__(self, path, loggername, tensors_names, tensor_nodes,
                 ds_handle, datasets_initializers, datasets_handles, datasets_eval_names,
                 erase_old=True):

        self._filename = os.path.join(path, loggername+".txt")
        self._plotfilename = os.path.join(path, loggername)

        reference_names = ["epoch", "dataset"]

        self._monitor_tensors_values, self._monitor_tensors_updates, self._monitor_tensors_reset = self._get_mean_ops(tensor_nodes)

        self._pd_csv_kwargs = {
            "sep" : "\t"
        }

        if (not erase_old) and os.path.exists(self._filename) and os.path.getsize(self._filename) > 0:
            self._df = pd.read_csv(self._filename, **self._pd_csv_kwargs)
        else:
            self._df = pd.DataFrame(columns=reference_names+tensors_names)

        os.makedirs(path, exist_ok=True)
        self._ds_handle = ds_handle
        self._datasets_initializers= datasets_initializers
        self._datasets_handles = datasets_handles
        self._datasets_eval_names = datasets_eval_names


    def log(self, sess, epoch):

        for ds_str in self._datasets_eval_names:
            monitor_tensors_values_np = evaluate_means_over_dataset(sess,
                                                                    self._ds_handle,
                                                                    self._datasets_initializers[ds_str],
                                                                    self._datasets_handles[ds_str],
                                                                    self._monitor_tensors_values,
                                                                    self._monitor_tensors_updates,
                                                                    self._monitor_tensors_reset)

            out = list(itertools.chain(monitor_tensors_values_np))
            new_series = pd.Series([epoch, ds_str]+out, index=self._df.columns)
            #NB as of today there is no `inplace` parameter for pandas dataframes
            self._df = self._df.append(new_series, ignore_index=True)

        self._df.to_csv(self._filename, index=False, float_format='%.6g', **self._pd_csv_kwargs)

    def reset(self, sess):
        _ = sess.run(self._monitor_tensors_reset)

    # def get_sess_run_args(self):
    #     return self._monitor_tensors_updates

    def plot(self, ylims=None, suffix=None, **kwargs):
        # subplots = False, sharex = True
        axes = self._df.plot(**kwargs)

        if ylims is not None:
            for ax, ylim in zip(axes, ylims):
                ax.set_ylim(ylim)

        self._finalize_plot(suffix=suffix)

    def plot_groupby(self, groupfield, suffix=None, **kwargs):
        ax=plt.gca()
        for label, df in self._df.groupby(groupfield):
            df.plot(**kwargs, ax=ax, label=groupfield+"_{:}".format(label))
        self._finalize_plot(suffix=suffix)

    def _finalize_plot(self, suffix=None):
        plt.tight_layout()
        plotfilename = self._plotfilename
        if suffix is not None:
            plotfilename+='_'+suffix
        plt.savefig(plotfilename+".png")
        plt.close()

    def _get_mean_ops(self, tensors, dataset_str=""):
        if dataset_str:
            dataset_str += "_"

        mean_v, mean_u_ops, mean_r_ops = \
                                zip(*[create_reset_metric(tf.metrics.mean,
                                                          scope=dataset_str + "mean_reset_metric/" + tnsr.name,
                                                          values=tnsr)
                                      for tnsr in tensors])
        return mean_v, mean_u_ops, mean_r_ops
