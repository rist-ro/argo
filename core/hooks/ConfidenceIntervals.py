from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.stats import chi2
import pygtc
import traceback
import tensorflow_probability as tfp
from scipy.optimize import curve_fit
from scipy.stats import beta
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error

class ConfidenceIntervals:

    def __init__(self,
                 dirName,
                 dataset_initializer,
                 dataset_handle,
                 handle,
                 n_samples_ph,
        #         session,
                 distr,
                 raw_x,
                 x,
                 y,
                 posterior_samples = 2500,
                 n_batches = -1):

        super().__init__()
#        self._datasets_keys= datasets_keys
        self._posterior_samples=posterior_samples
#        self._session=session
        self._dirName=dirName

        self._distsample = distr.sample()
        self._distmean = distr.mean()
        self._distvar = distr.variance()
        self._distcov = distr.covariance()

        self._raw_x=raw_x
        self._y=y
        self._x = x
        self._dataset_initializer=dataset_initializer
        self._dataset_handle=dataset_handle
        self._handle=handle
        self._n_batches=n_batches
        self._n_samples_ph = n_samples_ph

        #self.do_when_triggered(self._session,self._datasets_keys)



    def do_when_triggered(self,  session, datasets_keys):
        for ds_key in datasets_keys:
            fileName = "mc_"+str(ds_key)
            self._calculate_mc_dropout(session, ds_key)

    def _create_name(self, prefix, baseName):
       return self._dirName + "/" + prefix + '-' + baseName

    def CI_calibrate(self, total_covariance, mean_pred, rea_valu, baseName,alpha_calibrate=1,Aleatoric='Ale'):

        sumeT,ppf_run = self.general_ellip_counts_calibrate_core(total_covariance, mean_pred, rea_valu)
        sumeT_recali,ppf_run_recali = self.general_ellip_counts_calibrate_core(total_covariance, mean_pred, rea_valu,alpha_calibrate)

        fig_1 = plt.figure()
        plt.scatter(ppf_run,sumeT,label='uncalibrated')
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


    def _calculate_mc_dropout(self, session, ds_key, epoch):
        if type(session).__name__ != 'Session':
            raise Exception("I need a raw session to evaluate metric over dataset.")

        dataset_initializer = self._dataset_initializer[ds_key]
        dataset_handle = self._dataset_handle[ds_key]
        baseName=ds_key+"-ep{:d}".format(epoch)

        init_ops = dataset_initializer
        session.run(init_ops)

        count_68 = 0
        count_95 = 0
        count_99 = 0
        count_all = 0
        means_means = []
        covs_means = []
        means_covs = []
        Batch_samples_stack_T=[]
        Total_std = []
        Total_covariance=[]
        Real_valu=[]
        batch_means_A = []
        batch_reals_A = []
        batch_vars_A = []
        batch_covs_A = []
        batch_samples_A = []

        b = 0

        while True:

            batch_means = []
            batch_reals = []
            batch_vars = []
            batch_covs = []
            batch_samples = []


            try:
                # model.raw_x is the input before any noise addition (if present), we want to make sure we get the clean batch before noise
                batch_x, batch_y = session.run([self._raw_x, self._y],
                                               feed_dict={self._handle: dataset_handle})
                shape_batch=batch_y.shape[0]

                samples_A, means_A, varss_A, covs_A = session.run([self._distsample,
                                                              self._distmean,
                                                              self._distvar,
                                                              self._distcov],
                                              feed_dict={self._x: batch_x, self._n_samples_ph:1})
# samples_A, means_A, varss_A, covs_A = session.run([self._distsample, self._distmean, self._distvar, self._distcov], feed_dict={self._raw_x: batch_x, self._n_samples_ph:1})
                batch_means_A.append(means_A[:shape_batch,:])
                batch_vars_A.append(varss_A[:shape_batch,:])
                batch_covs_A.append(covs_A[:shape_batch,:,:])
                batch_samples_A.append(samples_A[:shape_batch,:])
                batch_reals_A.append(batch_y[:shape_batch,:])

                # model.x is the input after noise addition (if present), we want to make sure we feed x, so that noiose will not be added.
                for mcm in range(self._posterior_samples):
                    samples, means, varss, covs = session.run([self._distsample,
                                                              self._distmean,
                                                              self._distvar,
                                                              self._distcov],
                                                  feed_dict={self._x: batch_x,self._n_samples_ph:1})


                    batch_means.append(means[:shape_batch,:])
                    batch_vars.append(varss[:shape_batch,:])
                    batch_covs.append(covs[:shape_batch,:,:])
                    batch_samples.append(samples[:shape_batch,:])
                    batch_reals.append(batch_y[:shape_batch,:])

                batch_means_stack = np.stack(batch_means, axis=2)
                batch_vars_stack = np.stack(batch_vars, axis=2)
                batch_covs_stack = np.stack(batch_covs, axis=3)
                batch_samples_stack = np.stack(batch_samples, axis=2)

                coverage_value_68, coverage_value_95, coverage_value_99, coverage_all, \
                total_std, cov_pred_p, mean_covar_p, total_covariance, rea_valu, mean_pred = self.CI(
                                                        batch_means_stack,
                                                        batch_y,
                                                        batch_vars_stack,
                                                        batch_covs_stack,
                                                        baseName = baseName)

                # these are same for calibrated and uncalibrated
                means_means.append(mean_pred)
                covs_means.append(cov_pred_p)
                # this changes
                means_covs.append(mean_covar_p)
 #               cal_means_covs.append(cal_mean_covar_p)
                Batch_samples_stack_T.append(batch_samples_stack)

                Total_std.append(total_std)
                Total_covariance.append(total_covariance)
                Real_valu.append(rea_valu)

                count_68 += coverage_value_68
                count_95 += coverage_value_95
                count_99 += coverage_value_99
                count_all += coverage_all


                b+=1

                if b==self._n_batches:
                    break

            except tf.errors.OutOfRangeError:
                break


        np.save(self._create_name('cal_batch_means', baseName), batch_means_stack)
        np.save(self._create_name('cal_batch_covs', baseName), batch_covs_stack)
        np.save(self._create_name('cal_batch_reals', baseName), batch_reals)
        np.save(self._create_name('cal_batch_stack', baseName), batch_samples_stack)
        np.save(self._create_name('cal_batch_mean_covs', baseName), means_covs)
        np.save(self._create_name('cal_batch_covs_means', baseName), covs_means)
        np.save(self._create_name('cal_batch_means_means', baseName), means_means)
        np.save(self._create_name('cal_batch_Batch_samples_stack_T', baseName), Batch_samples_stack_T)
        means_means = np.stack(means_means, axis=0)
        Total_covariance = np.stack(Total_covariance, axis=0)

        Real_valu = np.stack(Real_valu, axis=0)

        cal_count_68, cal_count_95, cal_count_99,Total_batch,MSE = self.find_calibration( Total_covariance, means_means, Real_valu, baseName)
        #_, _, _,_ = self.find_calibration( batch_covs_A, batch_means_A, batch_reals_A, '_Ale_'+baseName)
        sumeT,ppf_run= self.general_ellip_counts_calibrate( batch_covs_A, batch_means_A, batch_reals_A,alpha_ini=1)
        fig_2 = plt.figure()
        plt.scatter(ppf_run,sumeT,label='uncalibrated-Ale')
        line_s1 = np.arange(0.0, 1, 0.01)
        plt.plot(line_s1, line_s1, 'r-', alpha=0.1)
        plt.xlabel('Confidence level')
        plt.ylabel('Estimated coverage probability')
        plt.legend()
        plt.savefig(self._create_name("calibration_Aleat", baseName) + ".png")
        plt.close(fig_2)

        with open(self._create_name('ci_info', baseName)+'.dat', 'w') as ft1:
            ft1.write("count_68 count_95 count_99 count_all MSE\n")
            #ft1.write("{} {} {} {} \n".format(count_68 , count_95, count_99, count_all))
            ft1.write("{} {} {} {} {}\n".format(cal_count_68 , cal_count_95, cal_count_99,Total_batch,MSE))


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

        # fig_1 = plt.figure()
        #
        # for i, param_name in enumerate(self._parameters_list):
        #     plt.errorbar(rea_valu[:, i], mean_pred[:, i], total_std[:, i], fmt='o', #color=colors[param_name], ecolor=ecolor[param_name],
        #                  elinewidth=3, capsize=0, label=param_name)
        #
        # line_s1 = np.arange(0.0, 1, 0.01)
        # plt.plot(line_s1, line_s1, 'r-', alpha=0.1)
        # plt.xlabel('True value')
        # plt.ylabel('Predicted value')
        # plt.legend()
        # plt.savefig(self._create_name("correlation", baseName) + ".png")
        # plt.close(fig_1)

        return sume68, sume95, sume99, sumeT,\
               total_std, cov_pred, mean_covar, total_covariance, rea_valu, mean_pred

    def ellip_counts_3_sigmas(self, covariance, mean, rea_values, alpha_calibrate):
        sume68 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=1)
        sume95 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=2)
        sume99 = self.general_ellip_counts(covariance, mean, rea_values, alpha_calibrate ,nstd=3)
        return sume68, sume95,sume99

    def find_calibration(self, covariance, means,real, baseName):
        shap=means.shape
        means_reshaped=means.reshape(shap[0]*shap[1],shap[2])
        real_reshaped=real.reshape(shap[0]*shap[1],shap[2])
        covariance_reshaped=covariance.reshape(shap[0]*shap[1],shap[2],shap[2])
        MSE=mean_squared_error(real_reshaped,means_reshaped)
        summe68, summe95,summe99 = self.ellip_counts_3_sigmas( covariance_reshaped, means_reshaped, real_reshaped, alpha_calibrate=1)
        self.CI_calibrate(covariance_reshaped, means_reshaped, real_reshaped, baseName,alpha_calibrate=1,Aleatoric='Epis')
        return summe68, summe95,summe99,shap[0]*shap[1],MSE


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
