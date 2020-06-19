import librosa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from argo.core.argoLogging import get_logger
from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook
from argo.core.utils.WavSaver import WavSaver

from core.wavenet.mel_utils_numpy import mfcc, mcd, mcd_per_sample
import seaborn as sns
import os

tf_logging = get_logger()


class AbstractWavHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dataset_keys,
                 hop_legth_cqt=128,
                 dirName=None,
                 **kwargs):

        super().__init__(model, period, time_reference, dataset_keys=dataset_keys, dirName=dirName, **kwargs)
        self._sample_rate = model.dataset.sample_rate

        if self._sample_rate is None:
            raise Exception("The sample rate is not set for the dataset")

        self.dir_name_anomaly_detection = dirName + '/anomaly_detection/'
        self.dir_name_reconstructions = dirName + '/reconstructed_wav/'

        self._wav_saver = WavSaver(self.dir_name_reconstructions, self._sample_rate)
        self.model_class_name = type(model).__name__
        self._hop_legth_cqt = hop_legth_cqt

        self._ds_initializers = model.datasets_initializers
        self._ds_handles_nodes = model.datasets_handles_nodes

        self.reconstr_metrics_file_names = {}

        for dir_ in [self.dir_name_reconstructions, self.dir_name_anomaly_detection]:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)

    def after_create_session(self, session, coord):
        '''

            initilizes self._samples and self._labels from datasets.
            previously it initilized them only with the
        '''
        super().after_create_session(session, coord)
        self._ds_handles = session.run(self._ds_handles_nodes)

        self._samples = {}
        self._labels = {}

        for (ds_key, index_list) in self._sample_indices_by_dataset.items():
            samples = self._model.dataset.get_elements(self._model.x, self._ds_handle,
                                                       self._ds_handles[ds_key],
                                                       self._ds_initializers[ds_key],
                                                       session,
                                                       None)  # passing None instead of index_list to return all samples
                                                            # indexing will be made in the hook because I need to
                                                            # calculate the reconstruction loss for generation

            labels = None
            if self._model.y is not None:
                labels = self._model.dataset.get_elements(self._model.y,
                                                          self._ds_handle,
                                                          self._ds_handles[ds_key],
                                                          self._ds_initializers[ds_key],
                                                          session,
                                                          None)

            labels = np.array([self._model.dataset.int_to_str_label(label) for label in labels])

            if index_list is None:
                index_list = list(range(0, samples.shape[0]))

            # return all samples and labels for a given ds_key and the corresponding indices
            self._samples[ds_key] = (index_list, samples)
            self._labels[ds_key] = (index_list, labels)

        # I set something like the following structure, e.g.
        # samples_dict = {TRAIN : (*indexes, [*examples]),
        #   VALIDATION : (*indexes, [*examples])
        # },

    def write_headers_to_reconstruction_files(self):
        # reconstr_metrics per label
        header = '#time_ref'
        unique_labels = set(self._labels['validation'][1])
        for metric in ['MCD', 'PSNR', 'loss']:
            for label in sorted(unique_labels):
                header += '\t{}_{} (std)'.format(metric, label)
            header += '\t{}_avg'.format(metric)
        header += '\n'

        for ds_key, file in self.reconstr_metrics_file_names.items():
            if not os.path.isfile(file):
                self.write_to_file(file, header)

    def _write_originals(self):
        for ds_key in self._samples:
            for index, sample in zip(*self._samples[ds_key]):
                self._wav_saver.save_wav(wav_data=sample,
                                         fileName="original_" + str(ds_key) + "_" + str(index))

    def plot_and_save(self, ds_key, indices, reconstructed_samples, samples, labels, time_ref, time_ref_str,
                      zs, prefix="", suffix=None, save_enc=False, plot_rainbowgram=False, save_wav=False):
        for index, sample, label, enc, rec_sample in zip(indices, samples, labels, zs, reconstructed_samples):
            fileName = prefix + str(ds_key) + "_" + str(index) + "_" + time_ref_str + "_" + str(
                time_ref).zfill(4) + (
                           "_" + suffix if suffix is not None else "")
            if save_enc:
                self._save_encoding(enc, fileName)

            if save_wav:
                self._wav_saver.save_wav(wav_data=rec_sample,
                                         fileName=fileName)
            self._plot_wav(sample, label, enc, rec_sample, fileName)

            if plot_rainbowgram:
                self._plot_wav_rainbowgram(sample, rec_sample, fileName + '_rainbowgram')

    def _save_encoding(self, enc, filename):
        np.save(self.dir_name_reconstructions + '/' + filename, enc)

    def _plot_wav(self, audio_orig, label, encoding, audio_rec, fileName):
        fig, axs = plt.subplots(3, 1, figsize=(15, 5))
        axs[0].plot(audio_orig)
        axs[0].set_title('Original Audio Signal - ' + str(label))
        axs[1].plot(encoding)
        axs[1].set_title('Encoding')
        axs[2].plot(audio_rec)
        axs[2].plot(audio_orig, alpha=0.5)
        axs[2].set_title('Reconstructed Audio Signal')
        plt.tight_layout()
        plt.savefig(self.dir_name_reconstructions + fileName + '.jpg', quality=90, format='jpg')
        plt.close()

    def _plot_wav_rainbowgram(self, audio_orig, audio_rec, fileName):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        cdict = {
            'red':   ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),
            'green': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),
            'blue':  ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),
            'alpha': ((0.0, 1.0, 1.0),
                      (1.0, 0.0, 0.0))
        }
        my_mask = LinearSegmentedColormap('MyMask', cdict)
        plt.register_cmap(cmap=my_mask)
        axs[0].set_title('Original Audio Signal')
        self._plot_magnitude_phase(audio_orig, axs[0], my_mask)
        axs[1].set_title('Reconstructed Audio Signal')
        self._plot_magnitude_phase(audio_rec, axs[1], my_mask)
        plt.savefig(self.dir_name_reconstructions + '/' + fileName + '.jpg', quality=90, format='jpg')
        plt.close()

    def _plot_magnitude_phase(self, samples, ax, mask, n_octaves=6, n_bins_per_octave=40, res_factor=0.8, peak=70.0):
        samples = np.squeeze(samples)
        samples = samples.astype(np.float32)
        # compute constant q transform
        C = librosa.cqt(samples,
                        sr=self._sample_rate,
                        hop_length=self._hop_legth_cqt,
                        bins_per_octave=n_bins_per_octave,
                        n_bins=n_octaves * n_bins_per_octave,
                        filter_scale=res_factor,
                        fmin=librosa.note_to_hz('C2'),
                        )

        # compute log magnitude and derivative of phase, see https://gist.github.com/jesseengel/e223622e255bd5b8c9130407397a0494
        mag, phase = librosa.core.magphase(C)
        phase_angle = np.angle(phase)
        phase_unwrapped = np.unwrap(phase_angle)
        dphase = phase_unwrapped[:, 1:] - phase_unwrapped[:, : -1]
        dphase = np.concatenate([phase_unwrapped[:, 0: 1], dphase], axis=1) / np.pi
        mag = (librosa.amplitude_to_db(mag, amin=1e-13, top_db=peak, ref=np.max) / peak) + 1
        # set colours
        ax.matshow(dphase[::-1, :], cmap=plt.cm.rainbow)
        # set intensities
        ax.matshow(mag[::-1, :], cmap=mask)
        ax.axis('off')

    # no plotting for EveryNEpochsImagesHook
    def plot(self):
        pass

    def multi_plot_and_save(self, ds_key, indices, reconstructed_samples_mean,
                            reconstructed_samples_sampling, samples, labels, time_ref, time_ref_str,
                            zs, means, covariances, prior_mean, prior_cov, prefix="", suffix=None, save_enc=False,
                            save_wav=False, plot_mean_separately=False, plot_rainbowgram=False):

        for index, sample, label, enc_sample, enc_mean, rec_sample_mean, rec_sample_sampling, covariance in \
                zip(indices, samples, labels, zs, means, reconstructed_samples_mean, reconstructed_samples_sampling, covariances):

            fileName = prefix + str(ds_key) + "_" + str(index) + "_" + time_ref_str + "_" + str(
                time_ref).zfill(4) + ("_" + suffix if suffix is not None else "")
            first_prefix_elem = prefix.split('_')[0]
            mean_suffix, sample_suffix = '_mean', '_sample'
            rec_file_name = fileName.replace(first_prefix_elem, 'reconstruction')

            if save_enc:
                file_name_encoding_sample = fileName.replace(first_prefix_elem, 'z_encoding') + sample_suffix
                file_name_encoding_mean = fileName.replace(first_prefix_elem, 'z_encoding') + mean_suffix
                self._save_encoding(enc_sample, file_name_encoding_sample)
                self._save_encoding(enc_mean, file_name_encoding_mean)

            if save_wav:
                self._wav_saver.save_wav(wav_data=rec_sample_mean, fileName=rec_file_name + mean_suffix)
                self._wav_saver.save_wav(wav_data=rec_sample_sampling, fileName=rec_file_name + sample_suffix)

            if plot_rainbowgram:
                self._plot_wav_rainbowgram(sample, rec_sample_mean, rec_file_name + mean_suffix + '_rainbowgram')
                self._plot_wav_rainbowgram(sample, rec_sample_sampling, rec_file_name + sample_suffix + '_rainbowgram')

            self._plot_wav_multi(sample, label, enc_sample, rec_sample_mean, rec_sample_sampling, enc_mean.T, fileName)

            if enc_mean.shape[0] > 2 and plot_mean_separately:
                mean_variance_file_name = fileName.replace(first_prefix_elem, 'mean_sigma')
                self._plot_mean_sigma(enc_mean, covariance, prior_mean, prior_cov, mean_variance_file_name)

    def spider_plot_encoding(self, ds_key, indices, samples, labels, time_ref, time_ref_shortstr,
                             zs, prefix, num_time_splits):
        for index, sample, label, z in zip(indices, samples, labels, zs):
            file_name = self.dir_name_reconstructions + '/' + prefix + str(ds_key) + "_" + str(index) + "_" + time_ref_shortstr + "_" + str(
                time_ref).zfill(4) + '.jpg'
            self._spider_plot(sample, z, num_time_splits, file_name, label)

    def _plot_wav_multi(self, audio_orig, label, encoding_sample, audio_rec_mean, audio_rec_sampling, mean, fileName):
        '''

        Args:
            means (np.array): the means of the samples, shape = (dim, channels)
            covariance (np.array): the covariance matrix of the samples associated with the mean,
                                    shape = (channels, dimension, dimension)

        '''
        fig, axs = plt.subplots(5, 1, figsize=(15, 12))
        axs[0].plot(audio_orig)
        axs[0].set_title('Original Audio Signal - ' + str(label))

        axs[1].plot(encoding_sample)
        axs[1].set_title('Encoding Sample')

        axs[2].plot(mean.T)
        axs[2].set_title('Encoding Mean')

        axs[3].plot(audio_rec_mean)
        axs[3].plot(audio_orig, alpha=0.5)
        axs[3].set_ylim([audio_orig.min(), audio_orig.max()])
        axs[3].set_title('Reconstructed Audio Signal from Mean')

        axs[4].plot(audio_rec_sampling)
        axs[4].plot(audio_orig, alpha=0.5)
        axs[4].set_ylim([audio_orig.min(), audio_orig.max()])
        axs[4].set_title('Reconstructed Audio Signal from Sample')

        plt.tight_layout()
        plt.savefig(self.dir_name_reconstructions + '/' + fileName + '.jpg', quality=90, format='jpg')
        plt.close()

    def _plot_mean_sigma(self, means, covariance, prior_means, prior_covs, file_name):
        '''
        plots the mean and variance (main diagonal of covariance matrix)
         of a multivariate gaussian distribution only for the first 2 channels
        Args:
            means (np.array): the means of the samples, shape=(latent_dim, channels)
            covariance (np.array): the covariance matrix of the samples associated with the mean,
                                    shape=(channels, latent_dim, latent_dim)
            prior_means (np.array): shape=(channels, latent_dim)
            prior_covs (np.array): shape=(channels, latent_dim, latent_dim)

        Returns:
            None
        '''
        sigmas = np.sqrt(np.diagonal(covariance, axis1=-1, axis2=-2))
        prior_sigmas = np.sqrt(np.diagonal(prior_covs, axis1=-1, axis2=-2))
        num_channels = means.shape[1]
        fig, axs = plt.subplots(num_channels, 1, figsize=(15, 12))
        for ax, mean, sigma, prior_mean, prior_sigma in zip(axs, means.T, sigmas, prior_means, prior_sigmas):
            self._filled_width_line_plot(ax, prior_mean, prior_sigma, 'r-', 'red', label='prior')
            self._filled_width_line_plot(ax, mean, sigma, 'b-', '#7f66ff', label='posterior')
            ax.set_ylim([(mean - sigma).min(), (mean + sigma).max()])
            ax.legend(loc=2, bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(self.dir_name_reconstructions + '/' + file_name + '.jpg', quality=90, format='jpg')
        plt.close()

    def _filled_width_line_plot(self, plot, x, tube_width, line_style, fill_color, label):
        '''
        plots the given x as a line and fills the space between [x - tube_width, x + tube_width] with the given fill color
        '''
        label = '{}: mu({:.3f}), sg({:.3f})'.format(label, np.mean(x), np.mean(tube_width))
        x_ticks = np.arange(0, x.shape[0])
        plot.plot(x_ticks, x, line_style, label=label)
        plot.fill_between(x_ticks, x - tube_width, x + tube_width, alpha=0.5, edgecolor=fill_color, facecolor=fill_color)

    def _spider_plot(self, x, z, num_time_splits, file_name, label):
        fig = plt.figure(constrained_layout=True, figsize=(20, 20))
        grid_spec = fig.add_gridspec(4, num_time_splits//2)

        split_signal_length = z.shape[0]//num_time_splits
        hoplen = x.shape[0] // z.shape[0]

        ax1_ = fig.add_subplot(grid_spec[0, :])
        ax1_.set_title('Original Signal - ' + str(label))
        ax1_.plot(x)

        ax2_ = fig.add_subplot(grid_spec[1, :])
        ax2_.set_title('Encoding Sample')
        ax2_.plot(z)

        x_coords_split_lines = np.arange(0, z.shape[0]+1, split_signal_length)
        for x_coord in x_coords_split_lines:
            ax2_.axvline(x=x_coord, c='k')
            ax1_.axvline(x=x_coord * hoplen, c='k')

        sub_zs = np.split(z, num_time_splits)

        plt.ylim(z.min(), z.max())
        for i, sub_z in enumerate(sub_zs):
            line = 2 if i < num_time_splits//2 else 3
            ax3_i = fig.add_subplot(grid_spec[line, i % (num_time_splits//2)], polar=True)
            ax3_i.set_title('Time Slice ' + str(i+1))

            self._spider(ax3_i, sub_z)

        plt.tight_layout()
        plt.savefig(file_name, format='jpg')
        plt.clf()

    def _spider(self, ax, z):
        length, channels = z.shape
        angles = [n / float(channels) * 2 * np.pi for n in range(channels)]
        angles += angles[:1]

        # x_labels
        plt.xticks(angles[:-1],  ['ch_' + str(ch+1) for ch in range(channels)])

        circular_z = np.concatenate([z, z[:, :1]], axis=1)
        for t in range(length):
            ax.plot(angles, circular_z[t, :], linestyle='solid')


    def write_to_file(self, file_name, string_):
        if not string_.endswith('\n'):
            string_ += '\n'
        with open(file_name, 'a+') as reconstr_metrics_file:
            reconstr_metrics_file.write(string_)

    def log_reconstr_loss(self, reconstr_loss, file_name):
        with open(file_name, 'r+') as reconstr_loss_file:
            lines = reconstr_loss_file.readlines()[1:]

        reconstr_losses = []
        global_steps = []
        for line in lines:
            line_split = line.strip().split('\t')
            global_steps.append(int(line_split[0]))
            reconstr_losses.append(float(line_split[1]))

        if self._time_ref not in global_steps:
            self.write_to_file(file_name,
                               string_='{}\t{:.2f}'.format(self._time_ref, reconstr_loss))
            plt.title('RL loss validation')
            plt.plot(global_steps, reconstr_losses)
            plt.savefig(file_name.replace('txt', 'jpg'))
            plt.clf()


    def is_model_vae(self):
        if self.model_class_name == 'WavenetVAE':
            return True
        elif self.model_class_name == 'WavenetAE':
            return False
        else:
            raise NotImplementedError('Please check function {} -- only support WavenetVAE and WavenetAE currently'
                                      .format(self.is_model_vae))

    def do_compute_reconstruction_metrics(self, x_original, x_reconstruction):
        # compute MCD
        sample_rate = self._model.dataset.sample_rate
        mfcc_original = mfcc(x_original, samplerate=sample_rate, preemph=0)
        mfcc_reconstructed = mfcc(x_reconstruction, samplerate=sample_rate, preemph=0)

        mse_per_sample = np.mean(np.square(x_original - x_reconstruction), axis=1)
        psnr_per_sample = 20 * np.log10(x_original.max()) - 10 * np.log10(mse_per_sample)

        return mcd_per_sample(mfcc_original, mfcc_reconstructed), psnr_per_sample.flatten()

    def log_reconstr_metrics(self, time_ref, samples, reconstr_samples, reconstr_loss, labels, file_name):
        # leave this sorted it is important to have the same format as the header
        unique_labels = sorted(set(labels))
        labels_arrays_dict = {}

        # we want to write in the format metric1_label1, metric1_label2, metric1_avg metric2_label1, metric_2_label2, .... etc.
        metrics = [[], [], []]
        for label in unique_labels:
            label_indices = np.argwhere(labels == label).flatten()
            mcd_, psnr = self.do_compute_reconstruction_metrics(samples[label_indices], reconstr_samples[label_indices])
            loss = reconstr_loss[label_indices]
            metrics[0].append(mcd_)
            metrics[1].append(psnr)
            metrics[2].append(loss)

        metric_names = ['mcd', 'psnr', 'loss']
        entry = str(time_ref)
        # fig, ax = plt.subplots(1, len(metric_names), figsize=(30, 25))

        for i, (metric_list, metric_name) in enumerate(zip(metrics, metric_names)):
            # ax[i].set_title(metric_name)
            for metric_per_label, label in zip(metric_list, unique_labels):
                # sns.kdeplot(metric_per_label, label=label, shade=True, ax=ax[i])
                entry += '\t{:.2f}({:.2f})'.format(np.mean(metric_per_label), np.std(metric_per_label))
            entry += '\t{:.2f}'.format(np.mean(np.concatenate(metric_list)))
        entry += '\n'

        # relative_file_name = file_name.split(os.path.sep)[-1]
        # relative_file_name_anomaly = self.anomaly_detection_dir + '/label_hist_' + relative_file_name.replace('.txt', '.png')
        # plt.savefig(relative_file_name_anomaly)
        # plt.clf()
        self.write_to_file(file_name, entry)
