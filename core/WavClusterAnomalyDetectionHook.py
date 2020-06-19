import os

from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook
import numpy as np
from datasets.Dataset import TEST
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class WavClusterAnomalyDetectionHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dataset_keys,
                 crops_per_sample,
                 sample_indices_by_dataset,
                 num_clusters=None,
                 dirName=None,
                 **kwargs):

        super().__init__(model, period, time_reference, dataset_keys=dataset_keys, dirName=dirName, **kwargs)
        self._ds_handles_nodes = model.datasets_handles_nodes
        self.crops_per_sample = crops_per_sample
        self.num_clusters = num_clusters
        self.sample_indices_by_dataset = sample_indices_by_dataset

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)
        self._ds_handles = session.run(self._ds_handles_nodes)
        self._samples = {}
        self._labels = {}

        for ds_key in self.sample_indices_by_dataset:
            x_test_crops = []
            y_test_labels = []
            for crop in range(self.crops_per_sample):                     #placeholder
                x_test = self._model.dataset.get_elements(self._model.x, self._ds_handle,
                                                           self._ds_handles[ds_key], # value la placeholld
                                                           self._ds_initializers[ds_key],
                                                           session,
                                                           None)  # passing None instead of index_list to return all samples
                                                                    # indexing will be made in the hook because I need to
                                                                    # calculate the reconstruction loss for generation
                x_test_crops.append(x_test)
                y_test = None
                if self._model.y is not None:
                    y_test = self._model.dataset.get_elements(self._model.y,
                                                              self._ds_handle,
                                                              self._ds_handles[ds_key],
                                                              self._ds_initializers[ds_key],
                                                              session,
                                                              None)

                # y_test = np.array([self._model.dataset.int_to_str_label(label) for label in y_test])
                y_test_labels.append(y_test)

            self._samples[ds_key] = np.concatenate(x_test_crops, axis=0)
            self._labels[ds_key] = np.concatenate(y_test_labels, axis=0)


    def do_when_triggered(self, run_context, run_values):
        for ds_key in self._samples:
            labels = self._labels[ds_key]
            encode_tuple = self._model.encode(self._samples[ds_key], sess=run_context.session)

            if len(encode_tuple) == 2:
                zs, x_shifted = encode_tuple
                hs = None
                covariance = None
            elif len(encode_tuple) == 6:
                zs, hs, covariance, prior_mean, prior_cov, x_shifted = encode_tuple
            else:
                raise ValueError("This tuple should not be this length: {}".format(len(encode_tuple)))

            config = '{}_{}'.format(self._samples[ds_key].shape[1], self.crops_per_sample)

            # plain
            self.cluster(zs, labels)

            # pca 3d
            zs_1d = zs.reshape([zs.shape[0], -1])
            pca = PCA(n_components=2)
            pca_transformed = pca.fit_transform(zs_1d)
            plt.figure()
            for label in np.unique(labels):
                idx_of_label = np.argwhere(labels == label).flatten()
                plt.scatter(x=pca_transformed[idx_of_label, 0], y=pca_transformed[idx_of_label, 1], label=str(label))

            plt.legend()
            plt.savefig(os.path.join(self._dirName, 'anomaly_detection', config + '_pca.png'))

            # t-SNE 3d
            tsne = TSNE(n_components=2, verbose=1, n_iter=5000)
            tsne_transformed = tsne.fit_transform(zs_1d)
            plt.figure()
            for label in np.unique(labels):
                idx_of_label = np.argwhere(labels == label).flatten()
                plt.scatter(x=tsne_transformed[idx_of_label, 0], y=tsne_transformed[idx_of_label, 1], label=str(label))

            plt.legend()
            plt.savefig(os.path.join(self._dirName, 'anomaly_detection', config + '_tsne.png'))

    def cluster(self, x, y):
        batch_size = x.shape[0]
        x = x.reshape([batch_size, -1])
        x = StandardScaler().fit_transform(x)

        self.num_clusters = self.num_clusters or len(np.unique(y))

        kmeans = KMeans(n_clusters=self.num_clusters, max_iter=1000)
        predicted_labels = kmeans.fit_predict(x)


        def change(arr, l1, l2):
            arr[arr == l1] = 3
            arr[arr == l2] = l1
            arr[arr == 3] = l2

        print(np.unique(predicted_labels, return_counts=True))
        print(classification_report(y, predicted_labels))
        print('Total:', sum(precision_score(y, predicted_labels, average=None)))
        print()


    def actual_labels(self, cluster_labels, y_actual):
        '''
        Associates most probable label with each cluster in KMeans model returns: dictionary of clusters
        assigned to each label
        '''
        # Initializing
        reference_labels = {}
        # For loop to run through each label of cluster label
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i, 1, 0)
            actual_label = np.bincount(y_actual[index == 1]).argmax()
            reference_labels[i] = actual_label

        return reference_labels, np.array(map(lambda x: reference_labels[x], cluster_labels))

    def log_result(self, report):
        pass