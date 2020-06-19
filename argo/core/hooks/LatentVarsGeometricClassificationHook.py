import tensorflow as tf
from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook
from ..argoLogging import get_logger

import numpy as np

from datasets.Dataset import TRAIN
from sklearn import linear_model, svm
import pdb

tf_logging = get_logger()


class LatentVarsGeometricClassificationHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 dirName,
                 tensors,
                 tensors_names,
                 datasets_keys,
                 period,
                 time_reference,
                 plot_offset=0,
                 extra_feed_dict={},
                 algorithm='svm',
                 kernel='rbf'
                 ):

        self._hook_name = "latent_vars_classifier_" + algorithm + "_" + kernel
        dirName = dirName + '/' + self._hook_name

        super().__init__(model,
                         period,
                         time_reference,
                         datasets_keys,
                         dirName=dirName,
                         plot_offset=plot_offset,
                         extra_feed_dict=extra_feed_dict)
        self._tensors = [[tensors]]
        self._tensors_names = [[tensors_names]]

        self._tensors_plots = [[{
            'fileName': "latent_vars_classifier",
            'logscale-y': 0}],
            ]
        self._tensors_values = {}

        self._fileName = "latent_vars_classifier"

        self._tensor_y_name = "y"
        self._tensor_y = tf.cast(self._model.y, dtype=tf.int64)
        self._algorithm = algorithm
        self._kernel = kernel

        tf_logging.info("Create LatentVarsGeometricClassificationHook")

    def _begin_once(self):
        pass

    def _get_encodings_and_labels(self, ds_key, tensor, session):
        dataset_initializer = self._ds_initializers[ds_key]

        session.run([dataset_initializer])
        encodings = []
        labels = []
        while True:
            try:
                batch_encodings, batch_labels = session.run([tensor, self._tensor_y],
                                                            feed_dict={self._ds_handle: self._ds_handles[ds_key]})
                encodings.append(batch_encodings)
                labels.append(batch_labels)
            except tf.errors.OutOfRangeError:
                break

        return np.concatenate(encodings), np.concatenate(labels)

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for LatentVarsGeometricClassificationHook - " + self._algorithm)

        session = run_context.session

        self._tensors_values = {}
        accuracies = {}

        latent_vars = {}
        labels = {}
        for (tensor, tensor_name) in zip(self._tensors[0][0], self._tensors_names[0][0]):
            for ds_key in self._datasets_keys:
                latent_vars[ds_key], labels[ds_key] = self._get_encodings_and_labels(ds_key, tensor, session)

            # train model
            if self._algorithm == 'svm':
                model = svm.SVC(kernel=self._kernel, gamma='auto')
            else:
                raise ValueError("Not implemented yet")
                # model = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='auto')
            model.fit(latent_vars[TRAIN], labels[TRAIN])

            for ds_key in self._datasets_keys:
                pred_latent_vars = model.predict(latent_vars[ds_key])
                accuracy = sum(pred_latent_vars == labels[ds_key]) / len(labels[ds_key])

                if not ds_key in accuracies.keys():
                    accuracies[ds_key] = []
                accuracies[ds_key].append(accuracy)

        for ds_key in self._datasets_keys:
            self._tensors_values[ds_key] = [[accuracies[ds_key]]]

        self.log_to_file_and_screen()

    def plot(self):
        super().plot()
