import lightning as pl
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast, get_linear_schedule_with_warmup

from config import Config


class BaseLMModel(pl.LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        config: Config
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.save_hyperparameters(ignore=["model", "tokenizer"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def on_save_checkpoint(self, checkpoint):
        """Save model and tokenizer with each checkpoint"""
        checkpoint["model_state"] = self.model.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """Load model state when loading checkpoint"""
        self.model.load_state_dict(checkpoint["model_state"])

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, logger=True, sync_dist=True)

        # Calculate perplexity
        perplexity = torch.exp(loss)
        self.log(
            "val/perplexity",
            perplexity,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss
