import numpy as np
import tensorflow as tf

from argo.core.argoLogging import get_logger
from datasets.Dataset import TRAIN

tf_logging = get_logger()

from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook

SUMMARIES_KEY = "3by3"

class ThreeByThreeHook(EveryNEpochsTFModelHook):

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 tensorboard_dir,
                 print_to_screen=True,
                 sample_size=10000,
                 dirName=None, fileName="truekl"):
        super().__init__(model, period, time_reference, [TRAIN], dirName=dirName,
                         tensorboard_dir=tensorboard_dir, trigger_summaries=False, plot_offset=0)
        self.sample_size = sample_size
        self._dirName = dirName

        self._print_to_screen = print_to_screen
        self._tensorboard_dir = tensorboard_dir

        self._tensors_to_average = None

        self._tensors_names = [[["uniform", "true-kl", "accuracy"]]]
        self._tensors_plots = [[{
            'fileName':   fileName,
            'logscale-y': 0}]]
        self._tensors_values = {}
        self._fileName = fileName

        self._x_flat = np.prod(self._model.dataset.x_shape)

    def begin(self):
        super().begin()

        self._define_true_likelihood()
        self._define_true_kl()

        self._register_summeries()

    def safe_loop_session_run(self, session, nodes, init_ops, feed_dict):

        if type(session).__name__ != 'Session':
            raise Exception("I need a raw session to evaluate metric over dataset.")

        result = None
        try:
            result = session.run(nodes, feed_dict={**feed_dict})
        except tf.errors.OutOfRangeError:
            session.run(init_ops)
            result = session.run(nodes, feed_dict={**feed_dict})
        return result

    def do_when_triggered(self, run_context, run_values):
        tf_logging.info("trigger for ThreeByThreeHook")

        true_kl, uniform_kl2, good_gen, bad_gen = self.safe_loop_session_run(
            run_context.session, [self.true_kl, self.uniform_kl2, self.good_gen, self.bad_gen],
            self._ds_initializers, feed_dict={
                self._model.n_z_samples: 1, self._model.b_size : self.sample_size})

        self._tensors_values[TRAIN] = [[[uniform_kl2, true_kl, good_gen]]]

        self.log_to_file_and_screen(log_to_screen=self._print_to_screen)

    def _define_true_likelihood(self):
        likelihood_pairs = {int("".join(str(int(x)) for x in pattern), 2): likelihood for pattern, likelihood
                            in self._model.dataset.likelihood}

        probs_true = np.zeros(2 ** self._x_flat)
        for i in likelihood_pairs.keys():
            probs_true[i] = likelihood_pairs[i]

        self.non_zeros = np.asarray([i > 0.0 for i in probs_true])
        self.probs_true = tf.constant(probs_true)

    def _define_true_kl(self):
        maximum_bin = 2 ** self._x_flat

        def binary_array_to_int(binary_array):
            return tf.reduce_sum(tf.cast(tf.reverse(tensor=binary_array, axis=[0]), dtype=tf.int32)
                                 * 2 ** tf.range(tf.cast(tf.size(binary_array), dtype=tf.int32)))

        gen_samples = tf.reshape(self._model.x_inferred, (self._model.n_z_samples * self._model.b_size, -1))
        gen_samples_zero_one = tf.cast(self._pm_cast(gen_samples), tf.int32)

        self.probs_gen = tf.bincount(tf.map_fn(binary_array_to_int, gen_samples_zero_one), minlength=maximum_bin,
                                     maxlength=maximum_bin)

        self.probs_gen = self.probs_gen / tf.reduce_sum(self.probs_gen)

        self.good_gen = tf.reduce_sum(tf.boolean_mask(self.probs_gen, self.non_zeros))
        self.bad_gen = tf.reduce_sum(tf.boolean_mask(self.probs_gen, tf.logical_not(self.non_zeros)))

        probs_true = self.probs_true

        self.true_kl = tf.reduce_sum(
            probs_true * (
                tf.log(tf.clip_by_value(self.probs_true, 1e-6, 1) / tf.clip_by_value(self.probs_gen, 1e-6, 1))))
        self.uniform_kl2 = tf.reduce_sum(probs_true * (
            tf.log(tf.clip_by_value(self.probs_true, 1e-6, 1) / np.asarray([1.0 / maximum_bin] * maximum_bin, dtype=np.float64))))

    def _register_summeries(self):
        for (name, node) in zip(self._tensors_names[0][0], [self.uniform_kl2, self.true_kl, self.good_gen]):
            self._register_summary_for_tensor(name, node)

    def _pm_cast(self, number):
        if self._model.dataset._pm_one:
            return (number+1)/2
        else:
            return number
