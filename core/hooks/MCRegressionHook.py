from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook
# get_samples_from_dataset
from datasets.Dataset import check_dataset_keys_not_loop, VALIDATION,TEST
from argo.core.argoLogging import get_logger
tf_logging = get_logger()
from .ConfidenceIntervalsOnlySamplesRegression import ConfidenceIntervalsOnlySamplesRegression


class MCRegressionHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName,
                 datasets_keys=[VALIDATION,TEST],
                 posterior_samples = 2500,
                 n_batches = -1
                 ):

        super().__init__(model, period, time_reference, dataset_keys=datasets_keys, dirName=dirName + '/mc_regression')
        self._default_plot_bool = False

        self._parameters_list = self._model.dataset._parameters_list
        self._n_batches = n_batches
        self._posterior_samples = posterior_samples

        tf_logging.info("Create MCRegressionHook for: \n" + \
                        ", ".join(datasets_keys)+"\n")

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)
        self.ci_obj = ConfidenceIntervalsOnlySamplesRegression(self._dirName,
                                                              self._ds_initializers,
                                                              self._ds_handles,
                                                              self._ds_handle,
                                                              self._model.n_samples_ph,
                                                              self._model.prediction_sample,
                                                              self._model.raw_x,
                                                              self._model.x,
                                                              self._model.y,
                                                              self._model.dataset._parameters_list,
                                                              posterior_samples=self._posterior_samples,
                                                              n_batches=self._n_batches)

    def do_when_triggered(self, run_context, run_values):
        time_ref = self._time_ref
        time_ref_str = self._time_ref_shortstr
        tf_logging.info("trigger for MCRegressionHook")
        self.ci_obj.do_when_triggered(run_context.session, self._datasets_keys, time_ref, time_ref_str)
        tf_logging.info("MCRegressionHook Done")

