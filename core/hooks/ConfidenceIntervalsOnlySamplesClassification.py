
from .ConfidenceIntervalsOnlySamples import ConfidenceIntervalsOnlySamples

import numpy as np

class ConfidenceIntervalsOnlySamplesClassification(ConfidenceIntervalsOnlySamples):

    def _stats_and_plot(self, baseName, batch_samples_list, real_valu_list, extra_batch_dict):

        all_samples = np.concatenate(batch_samples_list, axis=0)
        y = np.concatenate(real_valu_list, axis=0)

        nb, no, ns = all_samples.shape

        cumulative_preds = np.sum(all_samples, axis=2)

        predictions_forced = np.argmax(cumulative_preds, axis=1)
        accuracy_forced = np.mean(np.equal(predictions_forced, y)) * 100
        total = len(predictions_forced)

        #refuse prediction if uncertainties is too high (Bayesian defense)
        fracs = [0.5, 0.7, 0.9]
        accuracy_over_fracs = []
        for frac in fracs:
            accuracy_over_fracs.append(self._accuracy_over_threshold(cumulative_preds, frac, ns, y))

        with open(self._create_name("stats", baseName) + '.txt', 'w') as f:
            f.write("forced ->  accuracy: {:}  total: {:}\n".format(accuracy_forced, total))
            for frac, (acc_over, tot_over) in zip(fracs, accuracy_over_fracs):
                f.write("over {:} ->  accuracy: {:}  total: {:}\n".format(frac, acc_over, tot_over))


    def _accuracy_over_threshold(self, cumulative_preds, frac, ns, y):
        threshold = ns * frac
        more_than = np.array(cumulative_preds > threshold, dtype=np.int)
        non_zero_indexes = np.logical_not(np.all(more_than == 0, axis=1))
        predictions_more = np.argmax(more_than[non_zero_indexes], axis=1)
        y_more = y[non_zero_indexes]
        accuracy_more = np.mean(np.equal(predictions_more, y_more)) * 100
        return accuracy_more, len(predictions_more)

    #TODO what to plot for classification?
        # try:
        #     self._triangle_plot(these_samples, these_y, self._create_name("contours", baseName) + '.pdf')
        #     plt.close()
        # except Exception as e:
        #     print("ERROR: an Error occurred with plotGTC, continuing training... \n")
        #     print(traceback.format_exc())


