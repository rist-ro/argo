import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from ..utils.argo_utils import create_reset_metric


class LoggerHelper:

    def __init__(self, path, loggername, tensors_names, tensor_nodes):
        self._filename = os.path.join(path, loggername+".txt")
        self._plotfilename = os.path.join(path, loggername+".png")
        self._monitor_tensors_values, self._monitor_tensors_updates, self._monitor_tensors_reset = self._get_mean_ops(tensor_nodes)

        self._pd_csv_kwargs = {
            "sep" : "\t"
        }

        if os.path.exists(self._filename) and os.path.getsize(self._filename) > 0:
            self._df = pd.read_csv(self._filename, **self._pd_csv_kwargs)
        else:
            self._df = pd.DataFrame(columns=tensors_names)

        os.makedirs(path, exist_ok=True)


    def log(self, sess, epoch):
        monitor_tensors_values_np = sess.run(self._monitor_tensors_values)
        self._df.loc[epoch] = monitor_tensors_values_np
        self._df.to_csv(self._filename, **self._pd_csv_kwargs)

    def reset(self, sess):
        _ = sess.run(self._monitor_tensors_reset)

    def get_sess_run_args(self):
        return self._monitor_tensors_updates

    def plot(self):
        self._df.plot(subplots=True, sharex=True)
        plt.tight_layout()
        plt.savefig(self._plotfilename)
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

