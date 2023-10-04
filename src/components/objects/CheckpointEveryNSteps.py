import os
import pytorch_lightning as pl
from src.components.objects.Logger import Logger


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.last_save_global_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        """ Check if we should save a checkpoint after every train batch """
        if self.save_step_frequency is None:
            return
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0 and self.last_save_global_step != global_step:
            self.last_save_global_step = global_step
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            Logger.log(f"""CheckpointEveryNSteps checkpoint saved: {ckpt_path}.""",
                       log_importance=1)
