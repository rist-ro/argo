from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook
# get_samples_from_dataset
from datasets.Dataset import check_dataset_keys_not_loop, VALIDATION,TEST
from argo.core.argoLogging import get_logger
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
# from matplotlib.patches import Ellipse
from scipy.stats import chi2
import pygtc
import traceback
#import MCDcalibrationHook
#from ..hooks.MCDcalibrationHook import MCDcalibrationHook
#calibrated_number = MCDcalibrationHook()
import tensorflow_probability as tfp
from scipy.optimize import curve_fit
from scipy.stats import beta
# from matplotlib.patches import Ellipse
from scipy.stats import chi2
tf_logging = get_logger()


class MCDropoutHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName,
                 datasets_keys=[VALIDATION,TEST],
                 posterior_samples = 2500,
                 n_batches = -1
                 ):

        super().__init__(model, period, time_reference, dataset_keys=datasets_keys, dirName=dirName + '/mc_dropout')
        self._default_plot_bool = False

        self._parameters_list = self._model.dataset._parameters_list
        self._n_batches = n_batches
        self._posterior_samples = posterior_samples
        self.calibrated_value_Aleatoric = {}
        tf_logging.info("Create mcDropoutHook for: \n" + \
                        ", ".join(datasets_keys)+"\n")


    def do_when_triggered(self,  run_context, run_values):
        time_ref = self._time_ref
        time_ref_str = self._time_ref_shortstr
        tf_logging.info("trigger for mcDropoutHook")

        for ds_key in self._datasets_keys:
            fileName = "mc_"+str(ds_key) + "_" + time_ref_str + "_" + str(time_ref).zfill(4)
            self.calibrated_value_Aleatoric = self._calculate_mc_dropout_calibrate(run_context.session, ds_key, fileName)
            self._calculate_mc_dropout(run_context.session, ds_key, fileName)
            tf_logging.info("finished with %s"%ds_key)


    def _calculate_mc_dropout_calibrate(self, session, ds_key, baseName, is_training_value=False):
        if type(session).__name__ != 'Session':
            raise Exception("I need a raw session to evaluate metric over dataset.")

        dataset_initializer = self._ds_initializers[ds_key]
        dataset_handle = self._ds_handles[ds_key]
        handle = self._ds_handle
        model = self._model
        parameters_list = self._parameters_list


        # labels_min = self._labels_min
        # labels_max = self._labels_max
        # with open(self._create_name('max-min_info', baseName)+'.dat', 'w') as ft5:
        #     ft5.write("min_params  max_params\n")
        #     ft5.write("{} {} \n".format(labels_min, labels_max))


        init_ops = dataset_initializer
        session.run(init_ops)



        b = 0
        N_calibrated_batches=1000*self._n_batches
        batch_means = []
        batch_vars = []
        batch_covs = []
        batch_reals = []

        while True:



            try:
                # model.raw_x is the input before any noise addition (if present), we want to make sure we get the clean batch before noise
                batch_x, batch_y = session.run([model.raw_x, model.y],
                                               feed_dict={model.is_training: is_training_value,
                                                          handle: dataset_handle,
                                                          model.n_samples_ph:1})




                # model.x is the input after noise addition (if present), we want to make sure we feed x, so that noiose will not be added.


                samples, means, vars, covs = session.run([model.prediction_sample,
                                                              model.prediction_mean,
                                                              model.prediction_variance,
                                                              model.prediction_covariance],
                                              feed_dict={model.x: batch_x,
                                                         model.is_training: is_training_value,
                                                         handle: dataset_handle})

                batch_means.append(means)
                batch_vars.append(vars)
                batch_covs.append(covs)
                batch_reals.append(batch_y)

                b+=1

                if b==N_calibrated_batches:
                    break

            except tf.errors.OutOfRangeError:
                break

        # np.save(self._create_name('calibration_batch_means', baseName), batch_means)
        # np.save(self._create_name('calibration_batch_covs', baseName), batch_covs)
        # np.save(self._create_name('calibration_batch_reals', baseName), batch_reals)

        calibrated_value=self.calibrated_number(batch_covs[:-1], batch_means[:-1], batch_reals[:-1])
        # sumeT, ppf_run = self.CI_calibrate(
        #                         batch_covs,
        #                         batch_means,
        #                         batch_reals,
        #                         baseName,
        #                         alpha_calibrate=calibrated_value,
        #                         Aleatoric=1)
        # sumeT=np.array(sumeT)
        # ppf_run=np.array(ppf_run)
        # results_calibred=np.stack((sumeT,ppf_run)).T
        # np.save(self._create_name('ci_info_calibration', baseName), results_calibred)
        return calibrated_value




    def _create_name(self, prefix, baseName):
       return self._dirName + "/" + prefix + '_' + baseName

    def CI_calibrate(self, total_covariance, mean_pred, rea_valu, baseName,alpha_calibrate=1,Aleatoric=0):

        sumeT,ppf_run = self.general_ellip_counts_calibrate_core(total_covariance, mean_pred, rea_valu)
        sumeT_recali,ppf_run_recali = self.general_ellip_counts_calibrate_core(total_covariance, mean_pred, rea_valu,alpha_calibrate)

        fig_1 = plt.figure()
        plt.scatter(ppf_run,sumeT)
        plt.scatter(ppf_run_recali,sumeT_recali,label='calibrated')
        line_s1 = np.arange(0.0, 1, 0.01)
        plt.plot(line_s1, line_s1, 'r-', alpha=0.1)
        plt.xlabel('Confidence level')
        plt.ylabel('Estimated coverage probability')
        plt.legend()
        plt.savefig(self._create_name("calibration_{}".format(Aleatoric), baseName) + ".png")
        plt.close(fig_1)
        if Aleatoric:
            return sumeT, ppf_run

    def general_ellip_counts_calibrate(self, covariance, mean, real_values,alpha_ini=1):

        shapes=np.array(mean).shape
        shapes_batch=shapes[0]*shapes[1]
        means=np.array(mean).reshape(shapes_batch,shapes[2])
        reals=np.array(real_values).reshape(shapes_batch,shapes[2])
        covas=np.array(covariance).reshape(shapes_batch,shapes[2],shapes[2])
        return self.general_ellip_counts_calibrate_core(covas, means, reals,alpha_ini)

    def general_ellip_counts_calibrate_core(self, covas, means, reals,alpha_ini=1):
        shapes=np.array(means).shape
        Inverse_covariance = np.linalg.inv(covas)
        Ellip_eq = np.einsum('nl,nlm,mn->n', (reals - means), Inverse_covariance, (reals - means).T)
        ppf_run=list(np.arange(0.1, 1.0, 0.035))
        suma_T=[0] * len(ppf_run)
        rv = chi2(df=shapes[1])
        for ix, ppf in enumerate(ppf_run):
            square_norm = rv.ppf(ppf)
            values = Ellip_eq / (square_norm*alpha_ini)
            for ids, inst in enumerate(values):
                if inst <= 1:
                    suma_T[ix] += 1
                else:
                    pass
        return list(np.array(suma_T)/shapes[0]), list(ppf_run)

    def func(self,x, a, b):
        return  beta.cdf(x,a,b)
    def invfunc(self,x, a, b):
        return beta.ppf(x,a,b)


    def mininizer(self,datacov, datamean, datareal, x0, s):
        sumat_Re,ppft_Re=self.general_ellip_counts_calibrate(datacov, datamean, datareal,s)
        column_2_Re=np.array(sumat_Re)
        column_1_Re=np.array(ppft_Re)
        results_calibred_Re=np.stack((column_1_Re,column_2_Re)).T
        popt1_Re, pcov1_Re = curve_fit(self.func, results_calibred_Re[:,0],results_calibred_Re[:,1])
        return self.func(x0, *popt1_Re)

    def calibrated_number(self,datacov, datamean, datareal):
        x_val = np.linspace(0.2,1, 100)
        y_val = np.linspace(0.1,3, 50)
        summa=[]
        for s0 in y_val:
            summa.append(sum(np.abs(self.mininizer(datacov, datamean, datareal,x_val,s0)-x_val)))
        return y_val[np.argmin(summa)]



    def _calculate_mc_dropout(self, session, ds_key, baseName, is_training_value=False):
        if type(session).__name__ != 'Session':
            raise Exception("I need a raw session to evaluate metric over dataset.")

        dataset_initializer = self._ds_initializers[ds_key]
        dataset_handle = self._ds_handles[ds_key]
        handle = self._ds_handle
        model = self._model
        parameters_list = self._parameters_list


        init_ops = dataset_initializer
        session.run(init_ops)

        count_68 = 0
        count_95 = 0
        count_99 = 0
        count_all = 0
        cal_count_68 = 0
        cal_count_95 = 0
        cal_count_99 = 0
        cal_count_all = 0

        means_means = []
        covs_means = []
        cal_means_covs = []
        means_covs = []

        Total_std = []
        Total_covariance=[]
        cal_Total_std = []
        Real_valu=[]
        Batch_samples_stack_T=[]
        Batch_means_stack_T=[]
        Batch_covs_stack_T=[]

        b = 0

        while True:

            batch_means = []
            batch_vars = []
            batch_covs = []
            cal_batch_vars = []
            cal_batch_covs = []
            batch_samples = []

            try:
                # model.raw_x is the input before any noise addition (if present), we want to make sure we get the clean batch before noise
                batch_x, batch_y = session.run([model.raw_x, model.y],
                                               feed_dict={model.is_training: is_training_value,
                                                          handle: dataset_handle,
                                                          model.n_samples_ph:1})


                # model.x is the input after noise addition (if present), we want to make sure we feed x, so that noiose will not be added.
                for _ in range(self._posterior_samples):

                    samples, means, varss, covs = session.run([model.prediction_sample,
                                                              model.prediction_mean,
                                                              model.prediction_variance,
                                                              model.prediction_covariance],
                                              feed_dict={model.x: batch_x,
                                                         model.is_training: is_training_value,
                                                         handle: dataset_handle})
                    ##calibration###

                    cal_varss = varss#*calibrated_valued
                    cal_covs = covs#*calibrated_valued
#                     ssa=[]
#                     for i in range(means.shape[0]):
#                         ssa.append(np.random.multivariate_normal(means[i,:], cal_covs[i,:,:]))
#                     cal_samples = np.array(ssa)
#                     ####end

                    batch_means.append(means)
                    batch_vars.append(varss)
                    batch_covs.append(covs)
#                     cal_batch_vars.append(cal_varss)
#                     cal_batch_covs.append(cal_covs)
                    batch_samples.append(samples)##calibr

                batch_means_stack = np.stack(batch_means, axis=2)
                batch_vars_stack = np.stack(batch_vars, axis=2)
                batch_covs_stack = np.stack(batch_covs, axis=3)
#                 cal_batch_vars_stack = np.stack(cal_batch_vars, axis=2)
#                 cal_batch_covs_stack = np.stack(cal_batch_covs, axis=3)
                batch_samples_stack = np.stack(batch_samples, axis=2)

                coverage_value_68, coverage_value_95, coverage_value_99, coverage_all, \
                    total_std, cov_pred_p, mean_covar_p, total_covariance, rea_valu, mean_pred = self.CI(
                                                        batch_means_stack,
                                                        batch_y,
                                                        batch_vars_stack,
                                                        batch_covs_stack,
                                                        baseName = baseName)

#                 cal_coverage_value_68, cal_coverage_value_95, cal_coverage_value_99, cal_coverage_all, \
#                     cal_total_std, cal_cov_pred_p, cal_mean_covar_p, cal_total_covariance, cal_rea_valu, cal_mean_pred = self.CI(
#                                                         batch_means_stack,
#                                                         batch_y,
#                                                         cal_batch_vars_stack,
#                                                         cal_batch_covs_stack,
#                                                         baseName = "cal_"+baseName,
#                                                         alpha_calibrate=calibrated_valued)

                # these are same for calibrated and uncalibrated
                means_means.append(mean_pred)
                covs_means.append(cov_pred_p)
                # this changes
                means_covs.append(mean_covar_p)
 #               cal_means_covs.append(cal_mean_covar_p)

                Total_std.append(total_std)
                Total_covariance.append(total_covariance)
 #               cal_Total_std.append(cal_total_std)

                Real_valu.append(rea_valu)
                Batch_samples_stack_T.append(batch_samples_stack)
                Batch_means_stack_T.append(batch_means_stack)
                Batch_covs_stack_T.append(batch_covs_stack)

                count_68 += coverage_value_68
                count_95 += coverage_value_95
                count_99 += coverage_value_99
                count_all += coverage_all

#                 cal_count_68 += cal_coverage_value_68
#                 cal_count_95 += cal_coverage_value_95
#                 cal_count_99 += cal_coverage_value_99
#                 cal_count_all += cal_coverage_all

                b+=1

                if b==self._n_batches:
                    break

            except tf.errors.OutOfRangeError:
                break

        means_means = np.stack(means_means[:-1], axis=0)
        covs_means = np.stack(covs_means[:-1], axis=0)
        means_covs = np.stack(means_covs[:-1], axis=0)
        Total_std = np.stack(Total_std[:-1], axis=0)
        Total_covariance = np.stack(Total_covariance[:-1], axis=0)

#        cal_means_covs = np.stack(cal_means_covs, axis=0)
#        cal_Total_std = np.stack(cal_Total_std, axis=0)
        Real_valu = np.stack(Real_valu[:-1], axis=0)
        Batch_samples_stack = np.stack(Batch_samples_stack_T[:-1], axis=0)
        Batch_means_stack = np.stack(Batch_means_stack_T[:-1], axis=0)
        Batch_covs_stack = np.stack(Batch_covs_stack_T[:-1], axis=0)

        np.save(self._create_name('means_means', baseName), means_means)
        np.save(self._create_name('covs_means_', baseName), covs_means)
        np.save(self._create_name('means_covs_', baseName), means_covs)
        np.save(self._create_name('Total_std_', baseName), Total_std)
        #np.save(self._create_name('cal_means_covs_', baseName), cal_means_covs)
        #np.save(self._create_name('cal_Total_std_', baseName), cal_Total_std)
        np.save(self._create_name('Real_valu_', baseName), Real_valu)
        np.save(self._create_name('batch_samples_', baseName), Batch_samples_stack)
        np.save(self._create_name('Batch_means_stack_', baseName), Batch_means_stack)
        np.save(self._create_name('Batch_covs_stack_', baseName), Batch_covs_stack)

        cal_count_68, cal_count_95, cal_count_99, \
               calibrated_valued = self.find_calibration( Total_covariance, means_means, Real_valu,baseName)

        cal_samples= self.samples_calibrated(Batch_means_stack,Batch_covs_stack)
        np.save(self._create_name('cal_Batch_samples_stack', baseName), cal_samples)

        with open(self._create_name('ci_info', baseName)+'.dat', 'w') as ft1:
            ft1.write("count_68 count_95 count_99 count_all calibration\n")
            ft1.write("{} {} {} {} {}\n".format(count_68 , count_95, count_99, count_all, self.calibrated_value_Aleatoric))
            ft1.write("{} {} {} {} {}\n".format(cal_count_68 , cal_count_95, cal_count_99, count_all, calibrated_valued))



        # batch_samples_stack = np.array(list(map(lambda x,y: np.random.multivariate_normal(mean = x, cov = y), batch_means_stack, batch_covs_stack)))
        try:
            GTC = pygtc.plotGTC(chains=[np.transpose(batch_samples_stack[0])], figureSize=5, nContourLevels=2, sigmaContourLevels=True,
                                paramNames = parameters_list, plotName = self._create_name("fullGTC", baseName) + '.pdf')
            plt.close()
        except Exception as e:
            tf_logging.error(" an Error occurred with plotGTC, continuing training... \n")
            tf_logging.error(traceback.format_exc())

    def _create_name(self, prefix, baseName):
        return self._dirName + "/" + prefix + '_' + baseName

    def samples_calibrated(self,means_stack,covs_stack,alpha_calibrate=1):
        ssa1=[]
        shap=means_stack.shape
        shap_0=shap[0]*shap[1]
        means_stack_reshaped=means_stack.reshape(shap_0,shap[2],shap[3])
        covariance_stack_reshaped=covs_stack.reshape(shap_0,shap[2],shap[2],shap[3])
        covariance_stack_reshaped=covariance_stack_reshaped*alpha_calibrate
        for i in range(shap_0):
            ssa=[]
            for j in range(shap[3]):
                ssa.append(np.random.multivariate_normal(means_stack_reshaped[i,:,j], covariance_stack_reshaped[i,:,:,j]))
            ssa1.append(ssa)
        cal_samples = np.stack(ssa1, axis=2).T
        return cal_samples


    def CI(self, predictions, rea_valu, variance_s, covariance_s, baseName,alpha_calibrate=1):
        #  rea_valu=denormalize(rea_valu)
        batch_size = rea_valu.shape[0]

        mean_pred = np.mean(predictions, axis=2)
        var_pred = np.var(predictions, axis=2)
        # covariance over parameters only, for each example in the batch
        cov_pred = np.array(list(map(lambda x: np.cov(x), predictions)))
        mean_var = np.mean(variance_s, axis=2)
        mean_covar = np.mean(covariance_s, axis=3)

        total_variance = var_pred + mean_var
        total_covariance = cov_pred + mean_covar
        total_std = np.sqrt(total_variance)
        sume68,sume95, sume99 = self.ellip_counts_3_sigmas(total_covariance, mean_pred, rea_valu, alpha_calibrate)

        sumeT = batch_size #np.logical_and(rea_valu > confiden_inter_T_min, rea_valu < confiden_inter_T_max)

        fig_1 = plt.figure()

        for i, param_name in enumerate(self._parameters_list):
            plt.errorbar(rea_valu[:, i], mean_pred[:, i], total_std[:, i], fmt='o', #color=colors[param_name], ecolor=ecolor[param_name],
                         elinewidth=3, capsize=0, label=param_name)

        line_s1 = np.arange(0.0, 1, 0.01)
        plt.plot(line_s1, line_s1, 'r-', alpha=0.1)
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.legend()
        plt.savefig(self._create_name("correlation", baseName) + ".png")
        plt.close(fig_1)

        return sume68, sume95, sume99, sumeT,\
               total_std, cov_pred, mean_covar, total_covariance, rea_valu, mean_pred

    def ellip_counts_3_sigmas(self, covariance, mean, rea_values, alpha_calibrate):
        sume68 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=1)
        sume95 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=2)
        sume99 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=3)
        return sume68, sume95,sume99

    def find_calibration(self, covariance, means,real,baseName):
        shap=means.shape

        means_reshaped=means.reshape(shap[0]*shap[1],shap[2])
        real_reshaped=real.reshape(shap[0]*shap[1],shap[2])
        covariance_reshaped=covariance.reshape(shap[0]*shap[1],shap[2],shap[2])
        calibrated_value= self.calibrated_number(covariance, means, real)
        summe68, summe95,summe99 = self.ellip_counts_3_sigmas( covariance_reshaped, means_reshaped, real_reshaped, alpha_calibrate=calibrated_value)
        self.CI_calibrate(covariance_reshaped, means_reshaped, real_reshaped, baseName,alpha_calibrate=calibrated_value)
        return summe68, summe95,summe99,calibrated_value


    def general_ellip_counts(self, covariance, mean, real_values, alpha_calibrate=1,nstd=1):
        Inverse_covariance = np.linalg.inv(covariance)
        Ellip_eq = np.einsum('nl,nlm,mn->n', (real_values - mean), Inverse_covariance, (real_values - mean).T)
        if nstd == 1:
            ppf = 0.68
        if nstd == 2:
            ppf = 0.95
        if nstd == 3:
            ppf = 0.997

        rv = chi2(df=mean.shape[1])
        square_norm = rv.ppf(ppf)

        values = Ellip_eq / (square_norm*alpha_calibrate)
        suma_T = 0
        for ids, inst in enumerate(values):
            if inst <= 1:
                suma_T += 1
                # print(ids, inst)
            else:
                pass
        return suma_T
