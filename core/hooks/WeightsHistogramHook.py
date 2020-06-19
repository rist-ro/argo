from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook
from argo.core.argoLogging import get_logger
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf
import sonnet as snt

tf_logging = get_logger()

class WeightsHistogramHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 dirName):

        super().__init__(model, period, time_reference, dataset_keys=[], dirName=dirName + '/weights_histogram')

        self._default_plot_bool = False
        tf_logging.info("Create WeightsHistogramHook\n")

    def do_when_triggered(self,  run_context, run_values):
        time_ref = self._time_ref
        time_ref_str = self._time_ref_shortstr
        fileName = "Histogram_hook"+ "_" + time_ref_str + "_" + str(time_ref).zfill(4)
        self._calculate_histogram(run_context.session,  fileName)
        tf_logging.info("trigger for WeightsHistogramHook")

    def _plot_weight_posteriors(self, loc_post, untr_scale_post, loc_prior, untr_scale_prior, layer_names, figure_name):

        fig = plt.figure(figsize=(6, 3))

        ax = fig.add_subplot(2, 2, 1)
        for n, qm in zip(layer_names, loc_post):
            # try:
            #     sns.distplot(qm.flatten(), ax=ax)#, label=n)
            # except:
            sns.distplot(qm.flatten(), ax=ax, kde=False)

        ax.set_title("posterior weights mean")

        ax = fig.add_subplot(2, 2, 2)
        for n, qs in zip(layer_names, untr_scale_post):
            # qs_s=np.log(1.0 + np.exp(qs.flatten()))
            # try:
            #     sns.distplot(qs.flatten(), ax=ax)#, label=n)
            # except:
            sns.distplot(qs.flatten(), ax=ax, kde=False)

        ax.set_title("posterior weights untr scale")

        ax21 = fig.add_subplot(2, 2, 3)
        ax22 = fig.add_subplot(2, 2, 4)

        for n, loc, untr_sc  in zip(layer_names, loc_prior, untr_scale_prior):
            loc_flat = loc.flatten()
            sns.distplot(loc_flat, ax=ax21, kde=False)#, label=n)
            # scale_flat = np.log(1.0 + np.exp(np.asarray(untr_sc).flatten()))
            untr_scale_flat = np.asarray(untr_sc).flatten()
            if len(untr_scale_flat) == 1: # in case a layer has a single scale parameter (fixed prior)
                untr_scale_flat = np.tile(untr_scale_flat, [len(loc_flat)]) # I need to tile just for good histogram visualization
            sns.distplot(untr_scale_flat, ax=ax22, kde=False)#, label=n)

        ax21.set_title("prior weights mean")
        ax22.set_title("prior weights untr scale")

        lgd = plt.legend(labels=layer_names, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(self._create_name("Layers_", figure_name) + ".png", bbox_extra_artists=[lgd], bbox_inches='tight')
        plt.close(fig)

    def _create_name(self, prefix, baseName):
        return self._dirName + "/" + prefix + '_' + baseName

    def _calculate_histogram(self, session, baseName):
        if type(session).__name__ != 'Session':
            raise Exception("I need a  session to evaluate.")


        all_vars = self.get_weights(self._model._network)

        loc_posterior = [w for w in all_vars if "kernel_posterior_loc" in w.name]
        untr_scale_posterior = [w1 for w1 in all_vars if "kernel_posterior_untransformed_scale" in w1.name]

        loc_prior = [w2 for w2 in all_vars if "kernel_prior_loc" in w2.name]
        untr_scale_prior = [w3 for w3 in all_vars if "kernel_prior_untransformed_scale" in w3.name]

        _loc_post, _untr_scale_post, _loc_prior, _untr_scale_prior = session.run([loc_posterior,
                                                                       untr_scale_posterior,
                                                                       loc_prior,
                                                                       untr_scale_prior])

        layer_names = [os.path.basename(node.name) for node in loc_posterior]
        self._plot_weight_posteriors(_loc_post, _untr_scale_post, _loc_prior, _untr_scale_prior, layer_names, baseName)

    def get_weights(self, model):
        if isinstance(model, snt.AbstractModule):
            return model.get_all_variables()
        elif isinstance(model, tf.keras.Model):
            return model.variables
        else:
            raise Exception("what kind of model is this?")


