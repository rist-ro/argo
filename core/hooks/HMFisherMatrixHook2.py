import matplotlib
import numpy as np

from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.Dataset import TRAIN_LOOP, TRAIN

from argo.core.argoLogging import get_logger

tf_logging = get_logger()


def get_closest_to_square_divisor(number):
    sq = np.sqrt(number)
    nv = 1
    for i in range(int(sq), 0, -1):
        if number % i == 0:
            nv = i
            break
    return nv


def stich(M):
    nv = get_closest_to_square_divisor(M.shape[0])
    x = M.reshape([nv, -1, *M.shape[1:]])
    new_shape = x.shape
    pic = np.empty([x.shape[0] * x.shape[2], x.shape[1] * x.shape[3]])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            k = x.shape[2]
            l = x.shape[3]
            pic[i * k:i * k + k, j * l:j * l + l] = x[i, j, :, :]
    return pic, new_shape


def fix(M):
    if len(M.shape) == 1:
        M_m = M.reshape([-1, 1])
        shap = M_m.shape
    elif len(M.shape) == 2:
        M_m = M
        shap = M_m.shape
    else:
        M_m, shap = stich(M)
    return M_m, shap


class HMFisherMatrixHook2(EveryNEpochsTFModelHook):
    """
    Needed to Average certain variables every N steps
    """

    def __init__(self,
                 model,
                 dirName,
                 period,
                 train_loop_key=TRAIN_LOOP,
                 time_reference="epochs"
                 ):

        dirName = dirName + '/fisher'
        fileName = "fisher_matrix"
        average_steps = model._get_steps(period, time_reference)
        super().__init__(model=model, period=average_steps, time_reference=time_reference, dataset_keys=[TRAIN],
                         dirName=dirName, trigger_summaries=False,
                         plot_offset=0)

        self._saved_matrices = self._model._optimizer._saves

        values = list(self._saved_matrices.values()) + [self._model._optimizer._diagonal_pad]
        keys = list(self._saved_matrices.keys())
        self.layers = list(set([k.split("_")[0] for k in keys]))
        keys += ["d_pad"]

        self._tensors_to_average = [
            [[v] for v in values],
        ]

        self._tensors_names = [
            [[k] for k in keys],
        ]

        self._tensors_plots = [
            [{
                 "fileName": k} for k in keys]
        ]

        self._tensors_values = {}
        print("HMFisherMatrixHook2 has been enabled")

    def calc_values(self, run_context, run_values):

        run_context.session.run(self._ds_initializers)

        # mean over the other dataset_keys
        # for i, (tensors_vertical_panel, files_panel) in enumerate(zip(self._tensors_names,
        #                                                               self._tensors_plots)):
        #
        #     if len(tensors_vertical_panel) > 0:
        #
        #         # here it start the vertical panel
        #         for j, (tensors_names_panel, file_save) in enumerate(zip(tensors_vertical_panel, files_panel)):
        for dataset_str in self._datasets_keys:
            self._tensors_values[dataset_str] = run_context.session.run(
                self._tensors_to_average, feed_dict={
                    self._ds_handle: self._ds_handles[dataset_str]
                })

    def after_run(self, run_context, run_values):
        if self._trigged_for_step:
            tf_logging.info("trigger for HMFisherMatrixHook2")
            time_ref = run_context.session.run(self._time_reference_node)
            self._time_ref = self.cast_time_ref(time_ref)

            self.calc_values(run_context, run_values)

        super().after_run(run_context, run_values)

    def plot(self):
        for i, (tensors_vertical_panel, files_panel) in enumerate(zip(self._tensors_names,
                                                                      self._tensors_plots)):

            if len(tensors_vertical_panel) > 0:

                # here it start the vertical panel
                for j, (tensors_names_panel, file_save) in enumerate(zip(tensors_vertical_panel, files_panel)):

                    for dataset_str in self._datasets_keys:
                        filePath = self.get_file_path(file_save["fileName"])
                        d = self._tensors_values[dataset_str][i][j][0]
                        np.save(filePath, d)
                        if file_save["fileName"] != "d_pad":
                            filePath = self.get_file_path(file_save["fileName"], extra="step")
                            np.save(filePath, d)

                            m = np.asarray(d)

                            if len(m.shape) == 1:
                                plt.figure()
                                plt.hist(m, bins=100)
                                plt.savefig(filePath + '.png')
                            else:
                                m = m.ravel()
                                plt.figure()
                                plt.hist(m, bins=100)
                                plt.savefig(filePath + '.png')

                filePath = self.get_file_path("d_pad")
                d_pad = np.load(filePath + '.npy')
                for l in self.layers:
                    M_s = self.get_matrices(l)

                    if len(M_s.keys()) == 3:
                        bias_name = l + "_B"
                        bias = M_s[bias_name]
                        M = ((1.0 + d_pad) / d_pad) * bias
                        M = np.absolute(M)
                        self.plot_matrix(M, bias_name)

                        U_name = l + "_U"
                        U = M_s[U_name]

                        V_T = np.transpose(U, axes=[1, 0])
                        MII_name = l + "_MII"
                        MII = M_s[MII_name]

                        M2 = np.einsum('ij,ljk->lik', U, np.einsum('lij,jk->lik', MII, V_T))

                        M = d_pad * np.expand_dims(np.eye(M2.shape[-1]), axis=0) - M2

                        M = ((1.0 + d_pad) / d_pad) * M
                        M = np.absolute(M)
                        # Let's see
                        m = M.ravel()
                        filePath = self.get_file_path(l + "_MI")
                        plt.figure()
                        plt.hist(m, bins=100)
                        plt.savefig(filePath + '.png')

                        self.plot_matrix(M, l + "_INV")
                    elif len(M_s.keys()) == 2:

                        bias_name = l + "_B"
                        bias = M_s[bias_name]
                        M = ((1.0 + d_pad) / d_pad) * bias
                        M = np.absolute(M)

                        self.plot_matrix(M, bias_name)

                        mi_name = l + "_MI"
                        mi = M_s[mi_name]
                        M = np.absolute(mi)
                        self.plot_matrix(M, mi_name)

                    elif len(M_s.keys()) == 1:
                        bias_name = l + "_B"
                        bias = M_s[bias_name]
                        M = ((1.0 + d_pad) / d_pad) * bias
                        M = np.absolute(M)
                        self.plot_matrix(M, bias_name)
                    else:
                        raise ValueError("Invalid amount of matrices '{}'".format(M.keys()))

    def plot_matrix(self, M, name):
        M, shap = fix(M)
        filePath = self.get_file_path(name, ".png", extra="x".join(map(lambda x: str(x), shap)))
        plt.imsave(filePath, M, cmap="gray")

    def log_to_file_and_screen(self, log_to_screen=False):

        for i, (tensors_vertical_panel, files_panel) in enumerate(zip(self._tensors_names,
                                                                      self._tensors_plots)):

            if len(tensors_vertical_panel) > 0:
                # here it start the vertical panel
                for j, (tensors_names_panel, file_save) in enumerate(zip(tensors_vertical_panel, files_panel)):
                    for dataset_str in self._datasets_keys:
                        filePath = self.get_file_path(file_save["fileName"])
                        d = self._tensors_values[dataset_str][i][j][0]
                        np.save(filePath, d)

                        filePath = self.get_file_path(file_save["fileName"], extra="step")
                        d = self._tensors_values[dataset_str][i][j][0]
                        np.save(filePath, d)

    def _after_run(self, run_context, run_values):
        pass

    def _create_or_open_files(self):
        pass

    def _reset_file(self, session):
        pass

    def end(self, session):
        pass

    def get_file_path(self, name, type="", extra=""):
        return self._dirName + '/{}_'.format(name) + extra + "_" + self._time_reference_str[0] + str(
            self._time_ref).zfill(4) + '{}'.format(type)

    def get_matrices(self, layer):
        flat = np.asarray(self._tensors_names).reshape([-1])
        matrices = list(filter(lambda i: layer in i, flat))
        return {m: np.load(self.get_file_path(m, '.npy')) for m in matrices}
