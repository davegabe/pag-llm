import torch
import torch.nn.functional as F

from config import SentenceClassificationConfig
from pag_classification.baseline_model import BaselineClassifier, BatchType
from pag_classification.embeddings_dataset import SentenceEmbeddingsDataset


class PagScoreClassifier(BaselineClassifier):
    def __init__(self, cfg: SentenceClassificationConfig, train_dataset: SentenceEmbeddingsDataset):
        super().__init__(cfg)
        self.lambda_pag = cfg.lambda_pag
        self.lambda_ce = cfg.lambda_ce
        self.train_dataset = train_dataset

        self.n_classes = self.classifier.classifier[-1].weight.shape[0]
        self.classes_dict = self._create_classes_dict()

    def _forward_step(self, batch: BatchType, prefix_tag: str) -> torch.Tensor:
        x_embed, y_true = batch['embedding'], batch['label']

        with torch.inference_mode(False):  # Enable gradients again, even in validation or test mode
            n, d = x_embed.shape
            device = x_embed.device
            x_embed.requires_grad_(True)
            x_embed.retain_grad()

            output = self.classifier(x_embed)
            y_pred = torch.argmax(output, dim=1)

            loss_pag = torch.tensor(0., device=device)

            # FIXME: this is WRONG, but it works
            g_batch_z = torch.stack([
                self.r(n, j)
                for j in range(self.n_classes)
            ]) - x_embed

            for class_i in range(self.n_classes):
                # Find N random samples of this class
                g_batch_y = torch.ones(n, device=device, dtype=torch.int64) * class_i  # [N]

                # Compute the loss of the batches, with respect to this class
                loss_fixed_class = F.cross_entropy(output, g_batch_y)
                batch_z_grad_fixed_class = torch.autograd.grad(loss_fixed_class, [x_embed], create_graph=True)[0]

                # Compute the L_cos between grads of batch_z and g_y(batch_z)
                class_loss_pag = F.cosine_similarity(batch_z_grad_fixed_class, g_batch_z)
                class_loss_pag = 1 - torch.mean(class_loss_pag)

                # Accumulate over all classes
                loss_pag = loss_pag + class_loss_pag

        loss_pag = loss_pag / self.n_classes

        # Now, the NEW loss
        loss_ce = F.cross_entropy(output, y_true)
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

    def r(self, n_samples: int, class_i: int) -> torch.Tensor:
        """
        Given a batch X, return a sample of this class, for each sample in the batch.

        Args:
            n_samples: Number of samples to return.
            class_i: Number of the current class to fetch samples from.

        Returns:
            torch.Tensor of size [N, D] where D is the embedding dimension
        """
        # Extract N random indexes from this class
        all_class_indexes = self.classes_dict[class_i]

        indexes = torch.randperm(len(all_class_indexes))[:n_samples]
        class_indexes = all_class_indexes[indexes]

        samples_of_class_i = torch.stack([
            self.train_dataset[i]['embedding']
            for i in class_indexes
        ])

        return samples_of_class_i

    def _create_classes_dict(self) -> dict[int, torch.Tensor]:
        return {
            class_i: torch.tensor([
                i
                for i in range(len(self.train_dataset))
                if self.train_dataset[i]['label'] == class_i
            ])
            for class_i in range(self.n_classes)
        }
