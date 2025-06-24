import lightning as pl
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.common import compute_top_k_accuracies


class BaseLMModel(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerFast,
                 config: LLMPagConfig):
        super().__init__()
        self.model_name = model_name
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

    def _step(self, batch: BatchType, tag: str) -> torch.Tensor:
        outputs: CausalLMOutputWithPast = self.model(**batch.to_dict())

        loss = outputs.loss
        self.log(
            f"{tag}/loss_ce",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            f"{tag}/loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        # Calculate perplexity
        perplexity = torch.exp(loss)
        self.log(
            f"{tag}/perplexity",
            perplexity,
            prog_bar=False,
            logger=True,
            sync_dist=True
        )

        # Calculate accuracy
        n, t = batch.input_ids.shape
        v = self.tokenizer.vocab_size

        assert batch.shift_labels.shape == (n, t), \
            f"Expected batch.shift_labels to be of shape (n, t), but got {batch.shift_labels.shape}"
        target_first_labels = batch.shift_labels[batch.attention_mask == 1]

        assert outputs.logits.shape == (n, t, v), \
            f"Expected outputs.logits to be of shape (n, t, v), but got {outputs.logits.shape}"
        logits = outputs.logits[batch.attention_mask == 1]

        top_k_accuracies = compute_top_k_accuracies(target_first_labels, logits, k_samples=3, tag=tag)
        self.log_dict(
            top_k_accuracies,
            prog_bar=False,
            logger=True,
            sync_dist=True
        )

        return loss


    def training_step(self, batch: BatchType, batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: BatchType, batch_idx: int):
        return self._step(batch, "val")
