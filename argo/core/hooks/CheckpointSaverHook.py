import os

import tensorflow as tf

from .LoggingMeanTensorsHook import tf_logging

from tensorflow.train import SessionRunArgs

from ..utils.argo_utils import freeze_graph_create_pb

import pdb

class CheckpointSaverHook(tf.train.SessionRunHook):
    """
    In house CheckpointSaverHook, simple version.
    Saves checkpoints every N steps.
    """
    def __init__(self,
                 checkpoint_dir,
                 save_steps,
                 saver = None,
                 checkpoint_basename = "model.ckpt",
                 pb_output_nodes = ["logits"],
                 save_pb_at_end = False
                 ):
        """Initializes a `CheckpointSaverHook`.
        Args:
          checkpoint_dir: `str`, base directory for the checkpoint files.
          save_steps: `int`, save every N steps.
          saver: `Saver` object, used for saving.
          checkpoint_basename: `str`, base name for the checkpoint files.
        """

        tf_logging.info("Create CheckpointSaverHook")
        self._save_pb_at_end = save_pb_at_end
        self._pb_output_nodes = pb_output_nodes
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        self._save_steps = save_steps

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
          raise RuntimeError(
              "Global step should be created to use CheckpointSaverHook.")

    def after_create_session(self, session, coord):
        global_step = session.run(self._global_step_tensor)
        # self._save(session, global_step)

    def before_run(self, run_context):
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        # get the current value after train op. (you mean maybe the next value?)
        global_step = run_values.results + 1
        if global_step % self._save_steps == 0:
            self._save(run_context.session, global_step)

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        self._save(session, last_step)
        if self._save_pb_at_end:
            self._save_pb(session)

    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        tf_logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
        self._saver.save(session, self._save_path, global_step=step, write_meta_graph=False)

    def _save_pb(self, session):
        path_pb = os.path.join(self._checkpoint_dir, 'frozen_graph.pb')
        freeze_graph_create_pb(session,
                               output_names=self._pb_output_nodes,
                               variable_names_whitelist=None,
                               variable_names_blacklist=None,
                               output_filename = path_pb,
                               clear_devices = True)





            
