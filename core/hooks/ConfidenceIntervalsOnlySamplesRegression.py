from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import scipy
from getdist import plots, MCSamples

import traceback
from scipy.stats import chi2
from tqdm import tqdm
from .ConfidenceIntervalsOnlySamples import ConfidenceIntervalsOnlySamples


class ConfidenceIntervalsOnlySamplesRegression(ConfidenceIntervalsOnlySamples):

    def __init__(self,
                 dirName,
                 dataset_initializers,
                 dataset_handles,
                 handle,
                 n_samples_ph,
                 distsample,
                 raw_x,
                 x,
                 y,
                 parameters_list,
                 posterior_samples=2500,
                 n_batches=-1,
                 extra_nodes_to_collect = [],
                 extra_nodes_names = []
                ):

        super().__init__(dirName,
                         dataset_initializers,
                         dataset_handles,
                         handle,
                         n_samples_ph,
                         distsample,
                         raw_x,
                         x,
                         y,
                         posterior_samples=posterior_samples,
                         n_batches=n_batches,
                         extra_nodes_to_collect=extra_nodes_to_collect,
                         extra_nodes_names=extra_nodes_names)


        self._parameters_list = parameters_list


    def _stats_and_plot(self, baseName, batch_samples_list, real_valu_list, extra_batch_dict):
        these_samples = batch_samples_list[0][0].T
        these_y = real_valu_list[0][0]
        skewness = scipy.stats.skew(these_samples, axis=0)
        kurtosis = scipy.stats.kurtosis(these_samples, axis=0)
        with open(self._create_name("stats_contours", baseName) + '.txt', 'w') as f:
            f.write("skewness: {:}\n".format(skewness))
            f.write("kurtosis: {:}\n".format(kurtosis))

        try:
            self._triangle_plot(these_samples, these_y, self._create_name("contours", baseName) + '.pdf')
            plt.close()
        except Exception as e:
            print("ERROR: an Error occurred with plotGTC, continuing training... \n")
            print(traceback.format_exc())

    def _triangle_plot(self, these_samples, these_y, plotname):
        names = self._parameters_list
        labels = []

        level_lines = [0.2, 0.4, 0.6, 0.8, 0.95, 0.98]

        num_level_lines = len(level_lines)
        g = plots.getSubplotPlotter(width_inch=9)
        g.settings.num_plot_contours = num_level_lines

        mcsamples = MCSamples(samples=these_samples, names=names, labels=names)
        mcsamples.updateSettings({'contours': level_lines})

        g.triangle_plot([mcsamples], names,
                        # filled_compare=True,
                        legend_labels=labels,
                        legend_loc='upper right',
                        # filled=False,
                        contour_colors=['darkblue', 'green'],
                        #                     filled=True,
                        #                     contour_lws=[.2, .4, .68, .95, .98]
                        )

        n_params = len(names)

        for i in range(n_params):
            for j in range(n_params):
                if j > i:
                    continue

                ax = g.subplots[i, j]
                ax.axvline(these_y[j], color='black', ls='--', alpha=0.4)
                if i != j:
                    ax.axhline(these_y[i], color='black', ls='-.', alpha=0.4)

        g.export(plotname)


    # # AFTER HERE UNKNOWN, unused at the moment...
    # #TODO this should be modified to get CI from samples (probably 1D CIs... ?)
    # def CI_calibrate(self, total_covariance, mean_pred, rea_valu, baseName,alpha_calibrate=1,Aleatoric='Ale'):
    #
    #     sumeT,ppf_run = self.general_ellip_counts_calibrate_core(total_covariance, mean_pred, rea_valu)
    #     sumeT_recali,ppf_run_recali = self.general_ellip_counts_calibrate_core(total_covariance, mean_pred, rea_valu,alpha_calibrate)
    #
    #     fig_1 = plt.figure()
    #     plt.scatter(ppf_run,sumeT,label='uncalibrated')
    #     plt.scatter(ppf_run_recali,sumeT_recali,label='calibrated')
    #     line_s1 = np.arange(0.0, 1, 0.01)
    #     plt.plot(line_s1, line_s1, 'r-', alpha=0.1)
    #     plt.xlabel('Confidence level')
    #     plt.ylabel('Estimated coverage probability')
    #     plt.legend()
    #     plt.savefig(self._create_name("calibration_{}".format(Aleatoric), baseName) + ".png")
    #     plt.close(fig_1)
    #     if Aleatoric:
    #         return sumeT, ppf_run
    #
    # def general_ellip_counts_calibrate(self, covariance, mean, real_values,alpha_ini=1):
    #
    #     shapes=np.array(mean).shape
    #     shapes_batch=shapes[0]*shapes[1]
    #     means=np.array(mean).reshape(shapes_batch,shapes[2])
    #     reals=np.array(real_values).reshape(shapes_batch,shapes[2])
    #     covas=np.array(covariance).reshape(shapes_batch,shapes[2],shapes[2])
    #     return self.general_ellip_counts_calibrate_core(covas, means, reals,alpha_ini)
    #
    # def general_ellip_counts_calibrate_core(self, covas, means, reals,alpha_ini=1):
    #     shapes=np.array(means).shape
    #     Inverse_covariance = np.linalg.inv(covas)
    #     Ellip_eq = np.einsum('nl,nlm,mn->n', (reals - means), Inverse_covariance, (reals - means).T)
    #     ppf_run=list(np.arange(0.1, 1.0, 0.035))
    #     suma_T=[0] * len(ppf_run)
    #     rv = chi2(df=shapes[1])
    #     for ix, ppf in enumerate(ppf_run):
    #         square_norm = rv.ppf(ppf)
    #         values = Ellip_eq / (square_norm*alpha_ini)
    #         for ids, inst in enumerate(values):
    #             if inst <= 1:
    #                 suma_T[ix] += 1
    #             else:
    #                 pass
    #     return list(np.array(suma_T)/shapes[0]), list(ppf_run)
    #
    # def samples_calibrated(self,means_stack,covs_stack,alpha_calibrate=1):
    #     ssa1=[]
    #     shap=means_stack.shape
    #     shap_0=shap[0]*shap[1]
    #     means_stack_reshaped=means_stack.reshape(shap_0,shap[2],shap[3])
    #     covariance_stack_reshaped=covs_stack.reshape(shap_0,shap[2],shap[2],shap[3])
    #     covariance_stack_reshaped=covariance_stack_reshaped*alpha_calibrate
    #     for i in range(shap_0):
    #         ssa=[]
    #         for j in range(shap[3]):
    #             ssa.append(np.random.multivariate_normal(means_stack_reshaped[i,:,j], covariance_stack_reshaped[i,:,:,j]))
    #         ssa1.append(ssa)
    #     cal_samples = np.stack(ssa1, axis=2).T
    #     return cal_samples
    #
    #
    # def CI(self, predictions, rea_valu, variance_s, covariance_s, baseName, alpha_calibrate=1):
    #     #  rea_valu=denormalize(rea_valu)
    #     batch_size = rea_valu.shape[0]
    #
    #     mean_pred = np.mean(predictions, axis=2)
    #     var_pred = np.var(predictions, axis=2)
    #     # covariance over parameters only, for each example in the batch
    #     cov_pred = np.array(list(map(lambda x: np.cov(x), predictions)))
    #     mean_var = np.mean(variance_s, axis=2)
    #     mean_covar = np.mean(covariance_s, axis=3)
    #
    #     total_variance = var_pred + mean_var
    #     total_covariance = cov_pred + mean_covar
    #     total_std = np.sqrt(total_variance)
    #     sume68,sume95, sume99 = self.ellip_counts_3_sigmas(total_covariance, mean_pred, rea_valu, alpha_calibrate)
    #
    #     sumeT = batch_size #np.logical_and(rea_valu > confiden_inter_T_min, rea_valu < confiden_inter_T_max)
    #
    #     # fig_1 = plt.figure()
    #     #
    #     # for i, param_name in enumerate(self._parameters_list):
    #     #     plt.errorbar(rea_valu[:, i], mean_pred[:, i], total_std[:, i], fmt='o', #color=colors[param_name], ecolor=ecolor[param_name],
    #     #                  elinewidth=3, capsize=0, label=param_name)
    #     #
    #     # line_s1 = np.arange(0.0, 1, 0.01)
    #     # plt.plot(line_s1, line_s1, 'r-', alpha=0.1)
    #     # plt.xlabel('True value')
    #     # plt.ylabel('Predicted value')
    #     # plt.legend()
    #     # plt.savefig(self._create_name("correlation", baseName) + ".png")
    #     # plt.close(fig_1)
    #
    #     return sume68, sume95, sume99, sumeT,\
    #            total_std, cov_pred, mean_covar, total_covariance, rea_valu, mean_pred
    #
    # def ellip_counts_3_sigmas(self, covariance, mean, rea_values, alpha_calibrate):
    #     sume68 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=1)
    #     sume95 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=2)
    #     sume99 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=3)
    #     return sume68, sume95,sume99
    #
    # def find_calibration(self, covariance, means,real, baseName):
    #     shap=means.shape
    #     means_reshaped=means.reshape(shap[0]*shap[1],shap[2])
    #     real_reshaped=real.reshape(shap[0]*shap[1],shap[2])
    #     covariance_reshaped=covariance.reshape(shap[0]*shap[1],shap[2],shap[2])
    #     MSE=mean_squared_error(real_reshaped,means_reshaped)
    #     summe68, summe95,summe99 = self.ellip_counts_3_sigmas( covariance_reshaped, means_reshaped, real_reshaped, alpha_calibrate=1)
    #     self.CI_calibrate(covariance_reshaped, means_reshaped, real_reshaped, baseName,alpha_calibrate=1,Aleatoric='Epis')
    #     return summe68, summe95,summe99,shap[0]*shap[1],MSE
    #
    #
    # def general_ellip_counts(self, covariance, mean, real_values, alpha_calibrate=1,nstd=1):
    #     Inverse_covariance = np.linalg.inv(covariance)
    #     Ellip_eq = np.einsum('nl,nlm,mn->n', (real_values - mean), Inverse_covariance, (real_values - mean).T)
    #     if nstd == 1:
    #         ppf = 0.68
    #     if nstd == 2:
    #         ppf = 0.95
    #     if nstd == 3:
    #         ppf = 0.997
    #
    #     rv = chi2(df=mean.shape[1])
    #     square_norm = rv.ppf(ppf)
    #
    #     values = Ellip_eq / (square_norm*alpha_calibrate)
    #     suma_T = 0
    #     for ids, inst in enumerate(values):
    #         if inst <= 1:
    #             suma_T += 1
    #             # print(ids, inst)
    #         else:
    #             pass
    #     return suma_T
