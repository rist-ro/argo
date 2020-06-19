from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook
# get_samples_from_dataset
from datasets.Dataset import check_dataset_keys_not_loop, VALIDATION
from argo.core.argoLogging import get_logger
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

tf_logging = get_logger()


class CorrelationHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName,
                 datasets_keys = [VALIDATION],
                 n_samples = 10
                 ):

        self._dirName = dirName + '/correlation'
        super().__init__(model, period, time_reference, dataset_keys=datasets_keys, dirName=self._dirName)
        self._n_samples = n_samples
        self._default_plot_bool = False
        tf_logging.info("Create CorrelationHook for: \n" + \
                        ", ".join(datasets_keys)+"\n")

    def _begin_once(self):
        distr = self._model.prediction_distr
        mean, variance = self._mean_n_variance(distr.sample(self._n_samples), axis=0)
        self.prediction_mean = mean
        self.prediction_std = tf.sqrt(variance)

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)
        self._ds_handles = session.run(self._ds_handles_nodes)

    def do_when_triggered(self, run_context, run_values):
        time_ref = self._time_ref
        time_ref_str = self._time_ref_shortstr
        tf_logging.info("trigger for CorrelationHook")

        for ds_key in self._datasets_keys:
            fileName = self._dirName + "/" + "corrs_" + str(ds_key) + "_" + time_ref_str + "_" + str(time_ref).zfill(4)+".png"
            self._plot_correlation(run_context.session, ds_key, fileName)

        tf_logging.info("done")


    def _plot_correlation(self, session, ds_key, fileName):
        preds, stds, ys = self._evaluate_correlations_over_dataset(session,
                                                        self._model,
                                                        self._ds_handle,
                                                        self._ds_initializers[ds_key],
                                                        self._ds_handles[ds_key])

        params_numbers=ys.shape[1]
        fmt=['bo','ro','yo','go', 'po','co']
        for i in range(params_numbers):
            plt.errorbar(ys[:, i], preds[:, i], stds[:, i], fmt=fmt[i], label=ds_key+"{}".format(i))

        #plt.errorbar(ys[:, 0], preds[:, 0], varis[:, 0], fmt='bo', label=ds_key+"1")
        #plt.errorbar(ys[:, 1], preds[:, 1], varis[:, 1], fmt='ro', label=ds_key+"2")
        #plt.errorbar(ys[:, 2], preds[:, 2], varis[:, 2], fmt='yo', label=ds_key+"3")

        colors=['blue','red','green','orange', 'darkviolet','purple']
        ecolors=['lightblue','lightcoral','lightgreen','yellow', 'violet','magenta']

        # for m in range(3):
        #     plt.errorbar(ys[:,m], preds[:,m], varis[:,m],fmt='o', color=colors[m],
        #                          ecolor=ecolors[m], elinewidth=3, capsize=0, label=ds_key)

        plt.legend()
        plt.savefig(fileName)
        plt.close('all')


    def _evaluate_correlations_over_dataset(self, session, model, handle, dataset_initializer, dataset_handle, n_batches=3, feed_dict={}):
        if type(session).__name__ != 'Session':
            raise Exception("I need a raw session to evaluate metric over dataset.")

        init_ops = dataset_initializer
        session.run(init_ops)
        i = 0
        all_preds = []
        all_stds = []
        # all_covs = []
        all_ys = []
        while True:
            try:

                preds, stds, ys = session.run([self.prediction_mean, self.prediction_std, model.y],
                                                     feed_dict = {**feed_dict, handle: dataset_handle, model.n_samples_ph:1})

                # TODO HEctor if covs are not null, create the circles around the points
                # TODO (assume covs can be the covariances of the predictions and can be either diagonal matrioces or not)

                all_preds.append(preds)
                all_stds.append(stds)
                # all_covs.append(covs)
                all_ys.append(ys)
                i+=1
                if i>=n_batches:
                    break
            except tf.errors.OutOfRangeError:
                break
        all_preds = np.concatenate(all_preds, axis=0)
        all_ys = np.concatenate(all_ys, axis=0)
        # all_covs = np.concatenate(all_covs, axis=0)
        all_stds = np.concatenate(all_stds, axis=0)

        return all_preds, all_stds, all_ys

    def _mean_n_variance(self, input_tensor, axis=None, keepdims=False, name=None):
        name = name if name else "mean_n_variance"
        with tf.name_scope(name):
            means = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
            squared_deviations = tf.square(input_tensor - means)
            if not keepdims:
                means = tf.squeeze(means, axis=axis)

            return means, tf.reduce_mean(squared_deviations, axis=axis, keepdims=keepdims)

    def plot(self):
        pass
