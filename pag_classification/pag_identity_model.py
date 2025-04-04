import torch
import torch.nn.functional as F

from config import SentenceClassificationConfig
from pag_classification.baseline_model import BaselineClassifier, BatchType


class PagIdentityClassifier(BaselineClassifier):
    def __init__(self, cfg: SentenceClassificationConfig):
        super().__init__(cfg)
        self.lambda_pag = cfg.lambda_pag
        self.lambda_ce = cfg.lambda_ce

        self.n_classes = self.classifier.classifier[-1].weight.shape[0]

    def _forward_step(self, batch: BatchType, prefix_tag: str) -> torch.Tensor:
        x_embed, y_true = batch['embedding'], batch['label']

        with torch.inference_mode(False):  # Enable gradients again, even in validation or test mode
            x_embed.requires_grad_(True)
            x_embed.retain_grad()

            output = self.classifier(x_embed)
            y_pred = torch.argmax(output, dim=1)

            loss_ce = F.cross_entropy(output, y_true)
            grad_x = torch.autograd.grad(loss_ce, [x_embed], create_graph=True)[0]

            loss_pag = 1 - F.cosine_similarity(grad_x, x_embed).mean()

        # Now, the NEW loss
        # noinspection DuplicatedCode
        loss = self.lambda_ce * loss_ce + self.lambda_pag * loss_pag

        self.log_dict({
            f'{prefix_tag}/loss_ce': loss_ce,
            f'{prefix_tag}/loss': loss,
        })
        self.log(f'{prefix_tag}/loss_pag', loss_pag, prog_bar=True)

        accuracy = (y_pred == y_true).float().mean()
        self.log(f'{prefix_tag}/accuracy', accuracy.item(), on_epoch=True, prog_bar=True, on_step=False)

        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        # TODO: implement robustness evaluation too
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        # TODO: implement robustness evaluation too
        return super().test_step(batch, batch_idx)
