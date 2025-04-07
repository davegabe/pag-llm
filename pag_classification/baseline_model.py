from typing import NamedTuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import SentenceClassificationConfig
from pag_classification.embeddings_classifier import EmbeddingClassifier

BatchType = dict[str, torch.Tensor]


class PredictionOutput(NamedTuple):
    y_true: torch.Tensor
    y_pred: torch.Tensor
    logits: torch.Tensor


class BaselineClassifier(LightningModule):
    def __init__(self, cfg: SentenceClassificationConfig):
        super().__init__()
        self.classifier = EmbeddingClassifier(cfg.embedding_dim, cfg.num_classes)
        self.lr = cfg.learning_rate
        self.save_hyperparameters(ignore=['classifier'])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            threshold=1e-4,
            min_lr=1e-5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'monitor': 'train/loss',
            },
        }

    def forward(self, batch: BatchType | torch.Tensor):
        if not isinstance(batch, torch.Tensor):
            batch = batch['embedding']
        return self.classifier(batch)


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
        self.log(f'{prefix_tag}/accuracy', accuracy.item(), on_epoch=True, prog_bar=True, on_step=False)
        
        return loss

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'train')

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'val')

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'test')
