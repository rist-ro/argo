from argo.core.hooks.AbstractWavHook import AbstractWavHook
from argo.core.utils.WavSaver import WavSaver
from datasets.Dataset import check_dataset_keys_not_loop, VALIDATION, TRAIN
from argo.core.argoLogging import get_logger
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

tf_logging = get_logger()


class WavenetGaussianVisualizationHook(AbstractWavHook):
    '''
    Hook to plot the mean and the variance of the latent gaussian distribution
    '''

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName,
                 sample_indices_by_dataset={
                     VALIDATION: []},
                 hop_legth_cqt=128,
                 dataset_keys=[TRAIN, VALIDATION]
                 ):

        _dirName = dirName + '/mean_variance_plots'
        super().__init__(model, period, time_reference, dataset_keys=dataset_keys, hop_legth_cqt=hop_legth_cqt,
                         dirName=_dirName)

        self._sample_indices_by_dataset = sample_indices_by_dataset

        check_dataset_keys_not_loop(list(sample_indices_by_dataset.keys()))

        tf_logging.info("Create WavenetGaussianVisualizationHook for: \n"
                        + "\n".join([ds_key + ": " + ", ".join(map(str, idxs))
                                     for ds_key, idxs in sample_indices_by_dataset.items()]))

    def before_training(self, session):
        tf_logging.info("WavenetGaussianVisualizationHook")

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for WavenetGaussianVisualizationHook")

        for ds_key in self._samples:
            indices, samples = self._samples[ds_key]

            mean, covariance = self._model.get_mean_covariance(samples, sess=run_context.session)
            self.plot_mean_sigma(mean, covariance, ds_key, indices, self._time_ref, self._time_ref_shortstr,
                                    prefix='mean_sigma_', suffix='sample')
            tf_logging.info("finished with %s" % ds_key)

    def plot_mean_sigma(self, means, covariance, ds_key, indices, time_ref, time_ref_str, prefix='', suffix=None):
        '''
        plots the mean and variance (main diagonal of covariance matrix)
         of a multivariate gaussian distribution only for the first 2 channels
        Args:
            means (np.array): the means of the samples, shape=(batch_sz, dimension, channels)
            covariance (np.array): the covariance matrix of the samples associated with the mean,
                                    shape=(batch_sz, channels, dimension, dimension)
            ds_key (str): 'train' or 'validation'
            indices: the indices of the processed samples
            time_ref: the time step in which the hook has been triggered
            time_ref_str: 'ep' for epoch or other time step reference
            prefix (str): beginning of file name
            suffix: eof_name

        Returns:
            None
        '''
        sigmas = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
        for index, mean, sigma in zip(indices, means, sigmas):
            fileName = prefix \
                       + str(ds_key) \
                       + "_" + str(index) \
                       + "_" + time_ref_str \
                       + "_" + str(time_ref).zfill(4) \
                       + ("_" + suffix if suffix is not None else "")

            plt.title('Mu & Sigma of latent gaussian distribution')
            x = np.arange(0, mean.shape[0])
            sigma1, sigma2 = sigma[0], sigma[1]
            mean1, mean2 = mean[:, 0], mean[:, 1]

            plt.plot(x, mean1, 'r-', label='mean1')
            plt.plot(x, mean2, 'b-', label='mean2')

            plt.fill_between(x, mean1 - sigma1, mean1 + sigma1, alpha=0.5, edgecolor='#ff3842',
                             facecolor='#ff3842')  # red
            plt.fill_between(x, mean2 - sigma2, mean2 + sigma2, alpha=0.5, edgecolor='#1a009e',
                             facecolor='#7f66ff')  # blue
            plt.savefig(self._dirName + '/' + fileName + '.png')
            plt.legend()
            plt.close()
