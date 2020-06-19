import os

from argo.core.hooks.LoggingMeanTensorsHook import tf_logging
from argo.core.hooks.EveryNEpochsTFModelHook import EveryNEpochsTFModelHook


class CheckpointModelSaverHook(EveryNEpochsTFModelHook):
    """
    In house CheckpointSaverHook, save an Argo Model.
    Saves checkpoints every N steps.
    """

    def __init__(self,
                 model,
                 period,
                 time_reference,
                 checkpoint_basename="model.ckpt"):
        """
        Initializes a `CheckpointModelSaverHook`.

        Args:
            model: the Argo model
            period: the period for the hook
            time_reference: either epochs or steps
            checkpoint_basename: basename for model saving
        """

        super().__init__(model, period, time_reference)

        self._checkpoint_dir = model._checkpoint_dir
        self._saver = model._saver
        self._save_path = os.path.join(self._checkpoint_dir, checkpoint_basename)
        tf_logging.info("Create CheckpointSaverHook")

    def do_when_triggered(self, global_step, time_ref, run_context, run_values):
        self._save(run_context.session, time_ref)

    def end(self, session):
        time_ref = session.run(self._time_reference_node)
        self._save(session, time_ref)

    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        tf_logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
        self._saver.save(session, self._save_path, global_step=step, write_meta_graph=False)
