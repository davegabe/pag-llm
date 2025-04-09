from typing import NamedTuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import SentenceClassificationConfig
from pag_classification.embeddings_classifier import EmbeddingClassifier
from pag_classification.evaluation_metrics import evaluate_robustness, accuracy_fgsm

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
        self.test_dataset = None
        self.val_dataset = None
        self.save_hyperparameters(ignore=['classifier', 'test_dataset'])

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

    def _evaluate_adversarial_robustness(self, prefix_tag: str, dataset: torch.utils.data.Dataset):
        """Evaluate model robustness against different adversarial attacks on the full dataset"""
        # Set model to evaluation mode
        self.eval()

        # Enable gradient tracking since Lightning disables it by default during evaluation
        with torch.enable_grad():
            # APGD-CE attack (default epsilon)
            try:
                robustness = evaluate_robustness(self, attack_name='apgd-ce', dataset=dataset)
                self.log(f'{prefix_tag}/rob_apgd_ce', robustness, on_epoch=True)
            except Exception as e:
                print(f'Error during APGD-CE evaluation: {e}')
                self.log(f'{prefix_tag}/rob_apgd_ce', 0.0, on_epoch=True)
            
            # APGD-CE attack (epsilon=0.5)
            try:
                robustness = evaluate_robustness(self, attack_name='apgd-ce', eps=0.5, dataset=dataset)
                self.log(f'{prefix_tag}/rob_apgd_ce_eps_0_5', robustness, on_epoch=True)
            except Exception as e:
                print(f'Error during APGD-CE evaluation (eps=0.5): {e}')
                self.log(f'{prefix_tag}/rob_apgd_ce_eps_0_5', 0.0, on_epoch=True)
            
            # Square attack
            try:
                robustness = evaluate_robustness(self, attack_name='square', max_batches=10, dataset=dataset)
                self.log(f'{prefix_tag}/rob_square', robustness, on_epoch=True)
            except Exception as e:
                print(f'Error during Square attack evaluation: {e}')
                self.log(f'{prefix_tag}/rob_square', 0.0, on_epoch=True)
                
            # FGSM attacks at different alpha values
            fgsm_alphas = [1e-3, 5e-3, 1e-2]
            for alpha in fgsm_alphas:
                try:
                    clean_accuracy, adv_accuracy = accuracy_fgsm(
                        model=self,
                        alpha=alpha,
                        dataset=dataset
                    )
                    self.log(f'{prefix_tag}/fgsm_alpha_{alpha}_clean', clean_accuracy, on_epoch=True)
                    self.log(f'{prefix_tag}/fgsm_alpha_{alpha}_adv', adv_accuracy, on_epoch=True)
                except Exception as e:
                    print(f'Error during FGSM attack evaluation (alpha={alpha}): {e}')
                    self.log(f'{prefix_tag}/fgsm_alpha_{alpha}_clean', 0.0, on_epoch=True)
                    self.log(f'{prefix_tag}/fgsm_alpha_{alpha}_adv', 0.0, on_epoch=True)

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'train')

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'val')

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._forward_step(batch, 'test')
    
    def on_validation_epoch_end(self):
        """Run adversarial evaluation at the end of validation epoch"""
        if self.current_epoch > 0 and self.current_epoch % 50 == 0:
            self._evaluate_adversarial_robustness('val', self.val_dataset)
    
    # def on_test_epoch_end(self):
    #     """Run adversarial evaluation at the end of test epoch"""
    #     self._evaluate_adversarial_robustness('test', self.test_dataset)
