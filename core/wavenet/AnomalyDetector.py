from datasets.Dataset import VALIDATION, TEST
from .mel_utils_numpy import mfcc, mcd_per_sample
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PARAM_ANOMALY_LABELS = 'anomaly_labels'
PARAM_NORMAL_LABELS = 'normal_labels'
PARAM_GRID_SEARCH_SIZE = 'grid_search_size'
PARAM_NOTE = 'note'

LABEL_NORMAL = 'normal'
LABEL_ANOMALY = 'anomaly'


class AnomalyDetector:
    def __init__(self, anomaly_detection_params, anomaly_detection_dir, time_ref):
        self.anomaly_detection_params = anomaly_detection_params
        self.anomaly_labels = np.array(anomaly_detection_params[PARAM_ANOMALY_LABELS])
        self.normal_labels = np.array(anomaly_detection_params[PARAM_NORMAL_LABELS])
        self.grid_search_size = anomaly_detection_params[PARAM_GRID_SEARCH_SIZE]
        self.anomaly_detection_dir = anomaly_detection_dir
        self.time_ref = time_ref

        self.validation_samples = None
        self.validation_reconstructions = None
        self.validation_labels = None
        self.validation_reconstr_losses = None

        self.test_samples = None
        self.test_reconstructions = None
        self.test_labels = None
        self.test_reconstr_losses = None

    def set_data(self, ds_key, samples, reconstructions, labels, reconstruction_losses):
        if ds_key == VALIDATION:
            self.validation_samples = samples
            self.validation_reconstructions = reconstructions
            self.validation_labels = labels
            self.validation_reconstr_losses = reconstruction_losses

        elif ds_key == TEST:
            self.test_samples = samples
            self.test_reconstructions = reconstructions
            self.test_labels = labels
            self.test_reconstr_losses = reconstruction_losses

            self.n_anomalous_total_test = np.sum(np.sum((self.test_labels == self.anomaly_labels[:, None]).any(axis=0)))
            self.n_normal_total_test = np.sum(np.sum((self.test_labels == self.normal_labels[:, None]).any(axis=0)))

    def indices_of_labels(self, labels, target_labels):
        if target_labels is None:
            return np.arange(len(labels))

        return np.argwhere((labels == target_labels[:, None]).any(axis=0)).flatten()

    def compute_metrics(self, samples, reconstructions, reconstruction_losses, sample_rate, labels, target_labels):
        target_indices = self.indices_of_labels(labels, target_labels)
        target_samples = samples[target_indices]
        target_reconstructions = reconstructions[target_indices]

        # will have shape (bs, 1)
        mse = np.mean(np.square(target_samples - target_reconstructions), axis=1)
        psnr = 20 * np.log10(target_samples.max()) - 10 * np.log10(mse)

        mfcc_original = mfcc(target_samples, samplerate=sample_rate, preemph=0)
        mfcc_reconstructed = mfcc(target_reconstructions, samplerate=sample_rate, preemph=0)
        mcd = mcd_per_sample(mfcc_original, mfcc_reconstructed)

        return mcd, psnr.flatten(), reconstruction_losses[target_indices]

    def detect_anomalies(self, sample_rate, reconstruction_method):
        if self.test_labels is None:
            print('Skipping doing anomaly detection. Both validation and test must be available! Got only validation.')
        elif self.validation_samples is None:
            print('Skipping doing anomaly detection. Both validation and test must be available! Got only test.')

        # set threshhold from validation data
        mcd_validation, psnr_validation, loss_validation = self.compute_metrics(self.validation_samples,
                                                                                self.validation_reconstructions,
                                                                                self.validation_reconstr_losses,
                                                                                sample_rate,
                                                                                self.validation_labels,
                                                                                None,
                                                                                )

        binary_labels = np.array(list(map(self.label_binary_value, self.validation_labels)))
        self.pair_plots(mcd_validation, psnr_validation, loss_validation, binary_labels,
                        self.anomaly_detection_dir + '/pair_plot_metrics_{}_validation_{}.png'.format(reconstruction_method, self.time_ref))
        self.plot_3d(mcd_validation, psnr_validation, loss_validation, binary_labels,
                     self.anomaly_detection_dir + '/3Dplot_metrics_{}_validation_{}.png'.format(reconstruction_method, self.time_ref))

        # calculate test anomalies based on validation threshold

        mcd_test, psnr_test, loss_test = self.compute_metrics(self.test_samples,
                                                              self.test_reconstructions,
                                                              self.test_reconstr_losses,
                                                              sample_rate,
                                                              self.test_labels,
                                                              target_labels=None,
                                                              )

        binary_labels = np.array(list(map(self.label_binary_value, self.test_labels)))
        self.pair_plots(mcd_test, psnr_test, loss_test, binary_labels,
                        self.anomaly_detection_dir + '/pair_plot_metrics_{}_test_{}.png'.format(reconstruction_method, self.time_ref))
        self.plot_3d(mcd_test, psnr_test, loss_test, binary_labels,
                        self.anomaly_detection_dir + '/3Dplot_metrics_{}_test_{}.png'.format(reconstruction_method, self.time_ref))


        # grid search from the validation data to find best anomaly detection, based on std deviation of the metrics

        # mcd
        greater_than = lambda x, y: x >= y
        less_than = lambda x, y: x <= y

        best_scores_mcd = self.grid_search_threshold(mcd_test, greater_than, mcd_validation)
        best_scores_psnr = self.grid_search_threshold(psnr_test, less_than, psnr_validation)
        best_scores_loss = self.grid_search_threshold(loss_test, greater_than, loss_validation)

        all_metrics_scores = np.concatenate([best_scores_mcd, best_scores_psnr, best_scores_loss])

        data_frame = pd.DataFrame(all_metrics_scores,
                                  columns=['threshold', 'precision anomaly', 'precision normal', 'recall anomaly',
                                           'recall normal', 'discriminant power anomaly', 'youden index anomaly'])

        n_total_entries_per_metric = all_metrics_scores.shape[0] // 3
        data_frame['used_metric'] = ['MCD'] * n_total_entries_per_metric + ['PSNR'] * n_total_entries_per_metric \
                                    + ['LOSS'] * n_total_entries_per_metric
        data_frame = data_frame.set_index('used_metric')

        output_file = self.anomaly_detection_dir + '/anomaly_detection_{}_test_{}.csv'.format(reconstruction_method, self.time_ref)
        data_frame.to_csv(output_file, float_format='%.2f')

    def grid_search_threshold(self, metric_per_sample_test, compare_func, metric_per_sample_validation):
        true_anomaly_indices = self.indices_of_labels(self.test_labels, self.anomaly_labels)
        true_normal_indices = self.indices_of_labels(self.test_labels, self.normal_labels)

        assert len(np.intersect1d(true_anomaly_indices,
                                  true_normal_indices)) == 0, 'Got duplicates in normal and anomaly indices'
        # assert len(true_anomaly_indices) + len(true_normal_indices) == len(self.test_labels), 'Some labels are missing'

        scores = []
        std_metric_value = np.std(metric_per_sample_validation)
        mean_metric_value = np.mean(metric_per_sample_validation)
        for stddev in np.linspace(-3 * std_metric_value, std_metric_value * 3, self.grid_search_size):
            threshold = mean_metric_value + stddev
            anomaly_truth_table = compare_func(metric_per_sample_test, threshold)
            predicted_anomalous_indices = np.argwhere(anomaly_truth_table).flatten()
            predicted_normal_indices = np.argwhere(np.logical_not(anomaly_truth_table)).flatten()

            correctly_labeled_anomalous = len(np.intersect1d(predicted_anomalous_indices, true_anomaly_indices))
            correctly_labeled_normal = len(np.intersect1d(predicted_normal_indices, true_normal_indices))

            precision_anomaly = div(correctly_labeled_anomalous, len(predicted_anomalous_indices))
            recall_anomaly = div(correctly_labeled_anomalous, len(true_anomaly_indices))

            precision_normal = div(correctly_labeled_normal, len(predicted_normal_indices))
            recall_normal = div(correctly_labeled_normal, len(true_normal_indices))

            discriminant_power_anomaly = (np.sqrt(3) / np.pi) * (np.log10(div(recall_anomaly, (1 - recall_anomaly)))
                                                                 + np.log10(div(recall_normal, (1 - recall_normal))))

            youden_index = recall_anomaly - (1 - recall_normal)

            scores.append([threshold, precision_anomaly, precision_normal, recall_anomaly, recall_normal,
                           discriminant_power_anomaly, youden_index])

        return np.stack(scores, axis=0)

    def row_with_most_maxes_on_columns(self, matrix):
        maxes_on_rows = np.argmax(matrix, axis=0)
        vals, counts = np.unique(maxes_on_rows, return_counts=True)
        best_score_row = vals[np.argmax(counts)]

        return best_score_row

    def pair_plots(self, mcd, psnr, loss, labels, file_name):
        df = pd.DataFrame(
            {
                'mcd': mcd,
                'psnr': psnr,
                'loss': loss,
                'label': labels.flatten()
            }
        )

        plt.clf()
        g = sns.PairGrid(df, hue='label')
        g.map_upper(plt.scatter, s=1)
        g.map_lower(sns.kdeplot, shade=True, shade_lowest=False, n_levels=4, alpha=0.5)
        g.map_diag(sns.kdeplot, lw=1, legend=False, shade=True)
        g.add_legend()
        plt.savefig(file_name)

    def plot_3d(self, mcd, psnr, loss, labels, file_name):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        anomaly_indices = labels == LABEL_ANOMALY
        normal_indices = np.logical_not(anomaly_indices)
        ax.scatter(mcd[anomaly_indices], psnr[anomaly_indices], loss[anomaly_indices], color='orange', label=LABEL_ANOMALY)
        ax.scatter(mcd[normal_indices], psnr[normal_indices], loss[normal_indices], color='green', label=LABEL_NORMAL, marker='+')
        ax.set_xlabel('MCD')
        ax.set_ylabel('PSNR')
        ax.set_zlabel('LOSS')
        ax.legend()
        plt.savefig(file_name)
        plt.clf()

    def label_binary_value(self, label):
        if label in self.normal_labels:
            return LABEL_NORMAL
        else:
            return LABEL_ANOMALY

def div(x, y):
    if y == 0:
        return 1e-9
    return x / y
