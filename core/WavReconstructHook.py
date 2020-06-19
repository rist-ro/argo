from argo.core.hooks.AbstractWavHook import AbstractWavHook
from argo.core.utils.WavSaver import WavSaver
from datasets.Dataset import check_dataset_keys_not_loop, VALIDATION, TRAIN, TEST
from argo.core.argoLogging import get_logger
from .wavenet.utils import mu_law_numpy
from .wavenet.AnomalyDetector import AnomalyDetector
import numpy as np
import os

tf_logging = get_logger()


class WavReconstructHook(AbstractWavHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName,
                 sample_indices_by_dataset={
                     VALIDATION: []},
                 hop_legth_cqt=128,
                 dataset_keys=[TRAIN, VALIDATION],
                 save_wav=False,
                 reconstruct_from_mean=True,
                 batch_size=20,
                 _plot=True,
                 spider_plot_time_splits=None,
                 anomaly_detection_params=None,
                 compute_reconstruction_metrics=True
                 ):

        super().__init__(model, period, time_reference, dataset_keys=dataset_keys, hop_legth_cqt=hop_legth_cqt, dirName=dirName)

        self._plot = _plot
        self.reconstruct_from_mean = reconstruct_from_mean
        self.save_wav = save_wav
        self.batch_size = batch_size
        self.spider_plot_time_splits = spider_plot_time_splits
        self._sample_indices_by_dataset = sample_indices_by_dataset

        self.compute_reconstruction_metrics = compute_reconstruction_metrics

        self.anomaly_detection_params = anomaly_detection_params

        if compute_reconstruction_metrics:
            self.reconstr_metrics_file_names = {
                TRAIN: dirName + '/reconstr_metrics_tf_train.txt',
                VALIDATION: dirName + '/reconstr_metrics_tf_validation.txt',
                TEST: dirName + '/reconstr_metrics_tf_test.txt',
            }

        check_dataset_keys_not_loop(list(sample_indices_by_dataset.keys()))

        tf_logging.info("Create WavReconstructHook for: \n" + \
                        "\n".join([ds_key + ": " + ", ".join(map(str, idxs or ['all'])) \
                                   for ds_key, idxs in sample_indices_by_dataset.items()]))

    def before_training(self, session):
        tf_logging.info("WavReconstructHook, writing originals: " + str(self.save_wav))
        if self.save_wav:
            self._write_origanals()

        if self.compute_reconstruction_metrics:
            self.write_headers_to_reconstruction_files()

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for WavReconstructHook")

        anomaly_detector = None
        if self.anomaly_detection_params is not None:
            anomaly_detector = AnomalyDetector(self.anomaly_detection_params, self.dir_name_anomaly_detection,
                                               '{}{}'.format(self._time_ref_shortstr, self._time_ref))

        for ds_key in self._samples:
            indices, samples = self._samples[ds_key]
            _, labels = self._labels[ds_key]
            original_indices = indices

            samples = samples[indices]
            labels = labels[indices]
            indices = np.arange(0, samples.shape[0])

            encode_tuple = self._model.encode(samples, sess=run_context.session)

            if len(encode_tuple) == 2:
                zs, x_shifted = encode_tuple
                hs = None
                covariance = None
            elif len(encode_tuple) == 6:
                zs, hs, covariance, prior_mean, prior_cov, x_shifted = encode_tuple
            else:
                raise ValueError("This tuple should not be this length: {}".format(len(encode_tuple)))

            # compute quantized and feed below...
            x_target_quantized = mu_law_numpy(samples) + 128
            x_target_quantized = np.squeeze(x_target_quantized, axis=-1)

            reconstructed_samples, reconstr_loss_per_sample = self._model.decode_tf(zs, x_shifted,
                                                                         x_target_quantized=x_target_quantized,
                                                                         sess=run_context.session,
                                                                         batch_size=self.batch_size)

            if self.compute_reconstruction_metrics:
                self.log_reconstr_metrics(self._time_ref, samples, reconstructed_samples, reconstr_loss_per_sample,
                                          labels, self.reconstr_metrics_file_names[ds_key])

            if anomaly_detector is not None:
                anomaly_detector.set_data(ds_key, samples, reconstructed_samples, labels, reconstr_loss_per_sample)

            if self._plot:
                if hs is not None and covariance is not None and self.reconstruct_from_mean:  # if mean/covariance are not none -> VAE
                    reconstructed_samples_mean, reconstr_loss_mean = self._model.decode_tf(hs, x_shifted,
                                                                                           sess=run_context.session,
                                                                                           x_target_quantized=x_target_quantized,
                                                                                           batch_size=self.batch_size)

                    self.multi_plot_and_save(ds_key, original_indices, reconstructed_samples_mean[indices], reconstructed_samples[indices],
                                             samples[indices], labels[indices], self._time_ref, self._time_ref_shortstr,
                                             zs[indices], hs[indices], covariance[indices], prior_mean,
                                             prior_cov, prefix='multiplot_tf_', save_enc=False,
                                             save_wav=self.save_wav, plot_mean_separately=True, plot_rainbowgram=False)
                elif hs is None and covariance is None:  # AE
                    self.plot_and_save(ds_key, original_indices, reconstructed_samples[original_indices], samples[original_indices], labels[original_indices], self._time_ref, self._time_ref_shortstr,
                                       zs[original_indices], prefix="reconstruction_tf_", suffix="sample", save_enc=False, save_wav=self.save_wav)
            tf_logging.info("finished with %s" % ds_key)

        if anomaly_detector is not None:
            anomaly_detector.detect_anomalies(self._model.dataset.sample_rate, 'tf')
