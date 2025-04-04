from typing import NamedTuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import get_linear_schedule_with_warmup

from config import SentenceClassificationConfig
from pag_classification.embeddings_classifier import EmbeddingClassifier

BatchType = dict[str, torch.Tensor]


class PredictionOutput(NamedTuple):
    y_true: torch.Tensor
    y_pred: torch.Tensor
    logits: torch.Tensor


class BaselineClassifier(LightningModule):
    def __init__(self, config: SentenceClassificationConfig):
        super().__init__()
        self.classifier = EmbeddingClassifier(config.embedding_dim)
        self.lr = config.learning_rate

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }

    def predict_batch(self, batch: BatchType) -> PredictionOutput:
        x_embed, y_true = batch['embedding'], batch['label']
        output = self.classifier(x_embed)
        y_pred = torch.argmax(output, dim=1)
        return PredictionOutput(y_true, y_pred, output)

    def _forward_step(self, batch: BatchType, prefix_tag: str) -> torch.Tensor:
        y_true, y_pred, output = self.predict_batch(batch)

        loss = F.cross_entropy(output, y_true)
        self.log(f'{prefix_tag}/loss', loss, prog_bar=True, logger=True)

        accuracy = (y_pred == y_true).float().mean()
        self.log(f'{prefix_tag}/accuracy', accuracy.item(), on_epoch=True)

        return loss

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'train')

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'val')

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'test')
