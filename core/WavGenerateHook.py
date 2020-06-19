from argo.core.hooks.AbstractWavHook import AbstractWavHook
from datasets.Dataset import check_dataset_keys_not_loop, VALIDATION, TRAIN, TEST
from argo.core.argoLogging import get_logger
from .wavenet.AnomalyDetector import AnomalyDetector
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .wavenet.utils import mu_law_numpy

tf_logging = get_logger()


class WavGenerateHook(AbstractWavHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName,
                 sample_indices_by_dataset={
                     VALIDATION: []},
                 fast_gen=True,  # use fast_generation wavenet for reconstruction without teacher forcing
                 debug_fast_gen=True,
                 # use fast_generation wavenet with the true input shifted and quantized to reconstruct with teacher forcing and check the FastGen network
                 hop_legth_cqt=128,
                 dataset_keys=[TRAIN, VALIDATION],
                 save_wav=False,
                 compute_reconstruction_metrics=True,
                 _plot=True,
                 generate_from_mean=True,
                 spider_plot_time_splits=None,
                 anomaly_detection_params=None
                 ):

        super().__init__(model, period, time_reference, dataset_keys=dataset_keys, hop_legth_cqt=hop_legth_cqt,
                         dirName=dirName)

        self.model_class_name = type(model).__name__
        self.save_wav = save_wav
        self.compute_reconstruction_metrics = compute_reconstruction_metrics
        self._plot = _plot
        self.generate_from_mean = generate_from_mean
        self.spider_plot_time_splits = spider_plot_time_splits

        self.anomaly_detection_params = anomaly_detection_params

        if compute_reconstruction_metrics:
            self.reconstr_metrics_file_names = {
                TRAIN: dirName + '/reconstr_metrics_x_train.txt',
                VALIDATION: dirName + '/reconstr_metrics_x_validation.txt',
                TEST: dirName + '/reconstr_metrics_x_test.txt',
            }

        self._sample_indices_by_dataset = sample_indices_by_dataset

        self._fast_gen = bool(fast_gen)
        self._debug_fast_gen = bool(debug_fast_gen)

        check_dataset_keys_not_loop(list(sample_indices_by_dataset.keys()))

        tf_logging.info("Create WavGenerateHook for: \n" + \
                        "\n".join([ds_key + ": " + ", ".join(map(str, idxs or ['all'])) \
                                   for ds_key, idxs in sample_indices_by_dataset.items()]))

    def before_training(self, session):
        tf_logging.info("WavGenerateHook, writing originals: " + str(self.save_wav))
        if self.save_wav:
            self._write_originals()

        if self.compute_reconstruction_metrics:
            self.write_headers_to_reconstruction_files()

    def do_when_triggered(self, run_context, run_values):
        '''

        This works as follows:
        1. stack all samples (not just the ones given by the indices) from all ds_keys and generate their encoding
         which contains the ecnoding from all ds_keys and hs and zs stacked on the 0-axis
        2. from the stacked encoding reconstruct the samples
        3. de-stack the encodings and samples and compute reconstruction metrics and plot the specified samples by indices

        '''
        tf_logging.info("trigger for WavGenerateHook")

        combined_encodings, ds_keys, ds_key_batch_sizes, covariances, prior_means, prior_covs, x_shifted_list, x_target_combined = self.get_stacked_encodings(
            run_context)
        reconstructions_combined, reconstr_losses = self._model.decode(combined_encodings, input_qs=False,
                                                                       sess=run_context.session,
                                                                       x_target_quantized=x_target_combined)

        anomaly_detector = None
        if self.anomaly_detection_params is not None:
            anomaly_detector = AnomalyDetector(self.anomaly_detection_params, self.dir_name_anomaly_detection,
                                               '{}{}'.format(self._time_ref_shortstr, self._time_ref))

        start_batch = 0
        for ds_key, ds_key_batch_size, covariance, prior_mean, prior_cov, x_shifted in \
                zip(ds_keys, ds_key_batch_sizes, covariances, prior_means, prior_covs, x_shifted_list):
            indices, samples = self._samples[ds_key]
            _, labels = self._labels[ds_key]

            end_batch = start_batch + ds_key_batch_size
            zs = combined_encodings[start_batch:end_batch]
            reconstructed_samples = reconstructions_combined[start_batch:end_batch]
            reconstr_loss = reconstr_losses[start_batch:end_batch]

            if covariance is not None and self.generate_from_mean:
                # it means that we have stacked zs and hs for the current ds_key -> need to unstack
                zs, hs = np.split(zs, 2, axis=0)
                reconstructed_samples, reconstructed_samples_mean = np.split(reconstructed_samples, 2, axis=0)
                reconstr_loss, reconstr_loss_mean = np.split(reconstr_loss, 2, axis=0)
                reconstr_loss_mean = self._model.mean_reconstr_loss(reconstr_loss_mean)

            if self.compute_reconstruction_metrics:
                self.log_reconstr_metrics(self._time_ref, samples, reconstructed_samples, reconstr_loss, labels,
                                          self.reconstr_metrics_file_names[ds_key])

            if anomaly_detector is not None:
                anomaly_detector.set_data(ds_key, samples, reconstructed_samples, labels, reconstr_losses)

            if self.spider_plot_time_splits is not None:
                self.spider_plot_encoding(ds_key, indices, samples[indices], labels[indices],
                                          self._time_ref, self._time_ref_shortstr, zs[indices],
                                          prefix='spider_plot_', num_time_splits=self.spider_plot_time_splits)

            if self._plot and covariance is not None and self.generate_from_mean:
                self.multi_plot_and_save(ds_key, indices, reconstructed_samples_mean[indices],
                                         reconstructed_samples[indices], samples[indices], labels[indices],
                                         self._time_ref, self._time_ref_shortstr, zs[indices], hs[indices],
                                         covariance[indices], prior_mean, prior_cov,
                                         prefix='multiplot_x_', save_enc=False, save_wav=self.save_wav,
                                         plot_mean_separately=False, plot_rainbowgram=False)
            # AE
            elif self._plot:
                self.plot_and_save(ds_key, indices, reconstructed_samples[indices], samples[indices],
                                   labels[indices], self._time_ref, self._time_ref_shortstr, zs[indices],
                                   prefix="reconstruction_x_", suffix='sample', save_enc=False, save_wav=self.save_wav)

            start_batch = end_batch

            if self._debug_fast_gen:
                # TEST FOR DEBUGGING: FAST GENERATION AS TEACHER FORCING FOR COMPARISON
                reconstructed_samples = self._model.decode_debug_astf((hs if hs is not None else zs), x_shifted,
                                                                      sess=run_context.session)
                self.plot_and_save(ds_key, indices, reconstructed_samples[indices], samples[indices], self._time_ref,
                                   self._time_ref_shortstr,
                                   zs[indices], prefix="reconstruction_debug_astf_", suffix=None, save_enc=False,
                                   save_wav=self.save_wav)
            tf_logging.info("finished with %s" % ds_key)

        if anomaly_detector is not None:
            anomaly_detector.detect_anomalies(self._model.dataset.sample_rate, reconstruction_method='x')

    def get_stacked_encodings(self, run_context):
        '''
        stacks all the samples from all ds keys and returns the stacked resulting encoding on axis 0
        may have a structure like this if we also have VAE and encoding from the mean:

        0 axis stacking structure e.g. not specifically this order of train/validation:
        if self.generate_from_mean == True

        || zs_validation | hs_validation || zs_train | hs_train || ...

        else

        || zs_ds_validation  || zs_ds_train  || ...

        Returns:
            tensor (batches, time_length, channels) : the stacked encodings
            ds_keys (List): the order in whcih the ds_key samples are stacked in the stacked encoding tensor
            ds_key_batch_sizes (List): a list of integers indicating how much space each ds_key takes up
            covariances: the covariances corresponging to the means of the distribution. None if self.generate_from_mean == False
            x_shifted_list
            x_target_combined: the stacked target x for computing the reconstruction loss

        '''

        ds_keys = []
        ds_key_batch_sizes = []  # will store information about how much of the batch axis 0 is held by each ds_key samples
        stacked_encodings = None
        covariances = []
        prior_means = []
        prior_covs = []
        x_shifted_list = []
        x_target_combined = None

        for ds_key in self._samples:
            indices, samples = self._samples[ds_key]
            _, labels = self._labels[ds_key]

            ds_keys.append(ds_key)

            encode_tuple = self._model.encode(samples, sess=run_context.session)
            if len(encode_tuple) == 2:
                zs, x_shifted = encode_tuple
                hs, covariance, prior_mean, prior_cov = [None] * 4
            elif len(encode_tuple) == 6:
                zs, hs, covariance, prior_mean, prior_cov, x_shifted = encode_tuple
            else:
                raise ValueError("This tuple should not be this length: {}".format(len(encode_tuple)))
            covariances.append(covariance)
            prior_means.append(prior_mean)
            prior_covs.append(prior_cov)
            x_shifted_list.append(x_shifted)

            x_target_quantized = mu_law_numpy(samples) + 128
            x_target_quantized = np.squeeze(x_target_quantized, axis=-1)

            # store information about how much zs+hs of the current ds_key will be held in the combined encodings array
            ds_key_batch_size = zs.shape[0]
            encoding = zs
            if hs is not None and self.generate_from_mean:
                ds_key_batch_size *= 2
                encoding = np.concatenate([zs, hs], axis=0)
                x_target_quantized = np.concatenate([x_target_quantized, x_target_quantized], axis=0)

            ds_key_batch_sizes.append(ds_key_batch_size)

            if stacked_encodings is None:
                stacked_encodings = encoding
            else:
                stacked_encodings = np.concatenate([stacked_encodings, encoding])

            if x_target_combined is None:
                x_target_combined = x_target_quantized
            else:
                x_target_combined = np.concatenate([x_target_combined, x_target_quantized])

        return stacked_encodings, ds_keys, ds_key_batch_sizes, covariances, prior_means, prior_covs, x_shifted_list, x_target_combined
