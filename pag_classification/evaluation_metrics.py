from typing import Callable

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pag_classification import baseline_model


def attack_forward_pass_for(target_classifier: LightningModule) -> Callable[[torch.Tensor], torch.Tensor]:
    def _attack_forward_pass(input_nchw: torch.Tensor) -> torch.Tensor:
        # From NCHW to ND shape
        batch_size = input_nchw.size(0)
        input_nd = input_nchw.view(batch_size, -1)

        # Run the classifier on this input
        return target_classifier(input_nd)

    return _attack_forward_pass


def evaluate_robustness(model: LightningModule,
                        dataset: torch.utils.data.Dataset,
                        batch_size: int = 256,
                        max_batches: int = -1,
                        eps: float = 1e-3,
                        attack_name: str = 'apdg-ce',
                        verbose: bool = True) -> float:
    adversary = AutoAttack(
        attack_forward_pass_for(model),
        norm='L2',
        eps=eps,
        device=model.device,
        verbose=False,
        version='custom',
        attacks_to_run=[attack_name],
    )
    adversary.logger = None

    all_robustness = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataloader_iter = tqdm(dataloader, total=max_batches if max_batches >= 0 else None) if verbose else dataloader
    for i, batch in enumerate(dataloader_iter):
        batch: baseline_model.BatchType
        if i >= max_batches > -1:
            break

        embeddings_nd, labels = batch['embedding'].to(model.device), batch['label'].to(model.device)
        n, d = embeddings_nd.shape
        assert d == 768, 'Find a different NCHW shape for a D != 768'
        embeddings_nchw = embeddings_nd.view(n, 3, 16, 16)  # 3*16*16 = 768

        x_adv, y_adv = adversary.run_standard_evaluation(embeddings_nchw, labels, bs=n, return_labels=True)
        all_robustness += adversary.clean_accuracy(x_adv, labels, bs=n)

    all_robustness /= len(dataloader) if max_batches == -1 else max_batches
    return all_robustness


def accuracy_fgsm(model: LightningModule, dataset: Dataset, alpha: float) -> tuple[float, float]:
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    real_accuracy = torch.tensor(0.)
    adversarial_accuracy = torch.tensor(0.)

    for batch in dataloader:
        embeddings, labels = batch['embedding'].to(model.device), batch['label'].to(model.device)
        embeddings.requires_grad_(True)  # To apply FGSM

        output = model(embeddings)
        with torch.no_grad():
            classifications = output.argmax(dim=1)
            real_accuracy += (classifications == labels).float().sum().cpu()

        loss = F.cross_entropy(output, labels)

        embeddings_grad = torch.autograd.grad(loss, [embeddings], create_graph=True)[0]

        with torch.no_grad():
            adversarial = embeddings + alpha * embeddings_grad.sign()
            adversarial_out = model(adversarial)

            adversarial_classifications = adversarial_out.argmax(dim=1)
            adversarial_accuracy += (adversarial_classifications == labels).float().sum().cpu()

    real_accuracy /= len(dataset)
    adversarial_accuracy /= len(dataset)

    return real_accuracy.item(), adversarial_accuracy.item()
