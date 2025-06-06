import lightning as pl

from models.base_model import BaseLMModel


class ResetOptimizersWarmupCallback(pl.Callback):
    """
    Callback to reset optimizers and schedulers at the start of the first epoch after the warmup period.
    """

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: BaseLMModel) -> None:
        """
        Reset optimizers and schedulers at the start of the first epoch after the warmup period.
        """
        if trainer.current_epoch == pl_module.config.training.warmup_pretrain_epochs:
            print("\n\nResetting optimizers and schedulers after warmup period.\n")
            trainer.strategy.setup_optimizers(trainer)
