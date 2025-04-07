import torch
import torch.nn.functional as F

from config import SentenceClassificationConfig
from pag_classification.baseline_model import BaselineClassifier, BatchType
from pag_classification.embeddings_dataset import SentenceEmbeddingsDataset


class PagScoreClassifier(BaselineClassifier):
    def __init__(self, cfg: SentenceClassificationConfig, train_dataset: SentenceEmbeddingsDataset,
                 pag_similarity_dim: int, x_grad_sign: int):
        super().__init__(cfg)
        self.lambda_pag = cfg.lambda_pag
        self.lambda_ce = cfg.lambda_ce
        self.pag_samples_per_class = cfg.pag_samples_per_class
        self.train_dataset = train_dataset
        self.pag_similarity_dim = pag_similarity_dim
        self.x_grad_sign = x_grad_sign

        self.n_classes = self.classifier.classifier[-1].weight.shape[0]
        self.classes_dict = self._create_classes_dict()
        # self.train_mean, self.train_stddev = 0, 1  # TODO: self._compute_mean_stddev()

    def _forward_step(self, batch: BatchType, prefix_tag: str) -> torch.Tensor:
        x_embed, y_true = batch['embedding'], batch['label']

        # Normalize the embeddings
        # x_embed = (x_embed - self.train_mean) / self.train_stddev

        with torch.inference_mode(False):  # Enable gradients again, even in validation or test mode
            n, d = x_embed.shape
            k = self.pag_samples_per_class
            device = x_embed.device

            if x_embed.is_inference():
                # RuntimeError: Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.
                x_embed = x_embed.clone()

            x_embed.requires_grad_(True)
            x_embed.retain_grad()

            output = self.classifier(x_embed)
            repeated_output = output.repeat(k, 1)  # [K * N, C] = [N, C] repeated K times
            assert repeated_output.shape == (k * n, self.n_classes)

            loss_pag = torch.tensor(0., device=device)

            for class_i in range(self.n_classes):
                g_batch = self.r(k, class_i)
                # g_batch = (g_batch - self.train_mean) / self.train_stddev  # Normalize the target gradients
                assert g_batch.shape == (k, d)
                g_batch = g_batch.view(k, 1, d) - x_embed  # Broadcast: [k, 1, d] - [n, d]
                assert g_batch.shape == (k, n, d)

                g_batch = g_batch.view(k * n, d)  # [k * n, d]

                # Classify my X batch samples as the target class class_i
                class_target = torch.zeros(k * n, device=device, dtype=torch.int64) + class_i  # [k * n] class_i target
                class_pag_loss = F.cross_entropy(
                    repeated_output,
                    class_target,
                )
                assert class_pag_loss.ndim == 0  # It is a scalar

                # Compute the gradient of the loss with y=class_i (class_i may or may not be y_true) wrt the input
                batch_z_grad_fixed_class = torch.autograd.grad(
                    self.x_grad_sign * class_pag_loss,  # +/-1 * class_pag_loss
                    [x_embed],
                    create_graph=True,
                )[0].repeat(k, 1)
                assert batch_z_grad_fixed_class.shape == (k * n, d)

                assert g_batch.shape == batch_z_grad_fixed_class.shape
                loss_pag = loss_pag + 1 - F.cosine_similarity(batch_z_grad_fixed_class, g_batch,
                                                              dim=self.pag_similarity_dim)
                if self.pag_similarity_dim == 0:
                    assert loss_pag.shape == (d,), f'{loss_pag.shape=} != {(d,)}'
                elif self.pag_similarity_dim == 1:
                    assert loss_pag.shape == (k * n,), f'{loss_pag.shape=} != {(k * n,)}'
                else:
                    raise ValueError(f'Unknown pag similarity dim: {self.pag_similarity_dim}')

            loss_pag = loss_pag.mean() / self.n_classes

        loss_ce = F.cross_entropy(output, y_true)

        # Now, the NEW loss
        loss = self.lambda_ce * loss_ce + self.lambda_pag * loss_pag

        self.log_dict({
            f'{prefix_tag}/loss_ce': loss_ce,
            f'{prefix_tag}/loss': loss,
        })
        self.log(f'{prefix_tag}/loss_pag', loss_pag, prog_bar=True)

        y_pred = torch.argmax(output, dim=1)
        accuracy = (y_pred == y_true).float().mean()
        self.log(f'{prefix_tag}/accuracy', accuracy, on_epoch=True, prog_bar=True, on_step=False)

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

        return samples_of_class_i.to(self.device)

    def _create_classes_dict(self) -> dict[int, torch.Tensor]:
        return {
            class_i: torch.tensor([
                i
                for i in range(len(self.train_dataset))
                if self.train_dataset[i]['label'] == class_i
            ])
            for class_i in range(self.n_classes)
        }

    def _compute_mean_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and stddev of the training dataset.

        Returns:
            tuple: mean and stddev of the training dataset
        """
        train_embeddings = torch.stack([
            self.train_dataset[i]['embedding']
            for i in range(len(self.train_dataset))
        ])
        train_mean = train_embeddings.mean(dim=0)
        train_stddev = train_embeddings.std(dim=0)

        return train_mean, train_stddev


class PagScoreSimilarSamplesClassifier(PagScoreClassifier):
    def __init__(self, cfg: SentenceClassificationConfig, train_dataset: SentenceEmbeddingsDataset):
        super().__init__(cfg, train_dataset, pag_similarity_dim=1, x_grad_sign=-1)


class PagScoreSimilarFeaturesClassifier(PagScoreClassifier):
    def __init__(self, cfg: SentenceClassificationConfig, train_dataset: SentenceEmbeddingsDataset):
        super().__init__(cfg, train_dataset, pag_similarity_dim=0, x_grad_sign=1)

    def r(self, n_samples: int, _: int) -> torch.Tensor:
        # Ignore the class_i, and return n_samples random samples
        indexes = torch.randperm(len(self.train_dataset))[:n_samples]

        samples_of_class_whatever = torch.stack([
            self.train_dataset[i]['embedding']
            for i in indexes
        ])

        return samples_of_class_whatever.to(self.device)
