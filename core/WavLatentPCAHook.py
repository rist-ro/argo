from argo.core.hooks.AbstractWavHook import AbstractWavHook
from argo.core.utils.WavSaver import WavSaver
from datasets.Dataset import check_dataset_keys_not_loop, VALIDATION, TRAIN
from argo.core.argoLogging import get_logger
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

tf_logging = get_logger()


class WavLatentPCAHook(AbstractWavHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName,
                 sample_indices_by_dataset={
                     VALIDATION: []},
                 hop_legth_cqt=128,
                 dataset_keys=[TRAIN, VALIDATION],
                 target_dim=1
                 ):

        self._dir_name = dirName + '/pca_latent'
        super().__init__(model, period, time_reference, dataset_keys=dataset_keys, hop_legth_cqt=hop_legth_cqt, dirName=self._dir_name)

        self.model_class_name = type(model).__name__

        self._sample_indices_by_dataset = sample_indices_by_dataset

        check_dataset_keys_not_loop(list(sample_indices_by_dataset.keys()))

        tf_logging.info("Create WavLatentPCAHook for: \n" + \
                        "\n".join([ds_key + ": " + ", ".join(map(str, idxs)) \
                                   for ds_key, idxs in sample_indices_by_dataset.items()]))

    def before_training(self, session):
        tf_logging.info("WavLatentPCAHook before training")

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for WavLatentPCAHook")
        encode_function = self._model.encode_return_covariance if self.is_model_vae() else self._model.encode

        for ds_key in self._samples:
            indices, samples = self._samples[ds_key]

            encode_tuple = encode_function(samples, sess=run_context.session)

            if len(encode_tuple) == 2:
                zs, x_shifted = encode_tuple
                hs = None
                covariance = None
            elif len(encode_tuple) == 4:
                zs, hs, x_shifted, covariance = encode_tuple
            else:
                raise ValueError("This tuple should not be this length: {}".format(len(encode_tuple)))

            # zs shape = (bs, time_len, channels)
            # write_file = '{}/pca_1_dim_{}_{}.txt'.format(self._dir_name, self._time_ref)
            pca_zs = []
            for zs_i in zs:
                pca_zs_i = PCA(n_components=1).fit_transform(zs_i)
                pca_zs.append(np.squeeze(pca_zs_i))
                plt.plot(pca_zs_i, '-')

            plt.savefig('{}/pca_1_dim_{}_{}_wave.jpg'.format(self._dir_name, self._time_ref, ds_key))

            if len(pca_zs) > 1:
                pca_zs = np.stack(pca_zs)
                pca_zs = PCA(n_components=2).fit_transform(pca_zs)
                plt.clf()
                plt.plot(pca_zs[:, 0], pca_zs[:, 1], '.')
                plt.savefig('{}/pca_1_dim_{}_{}_points.jpg'.format(self._dir_name, self._time_ref, ds_key))

            tf_logging.info("finished with %s" % ds_key)

    def is_model_vae(self):
        if self.model_class_name == 'WavenetVAE':
            return True
        elif self.model_class_name == 'WavenetAE':
            return False
        else:
            raise NotImplementedError('Please check function {} -- only support WavenetVAE and WavenetAE currently'
                                      .format(self.is_model_vae))

