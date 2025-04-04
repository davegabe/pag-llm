import torch.nn as nn

from torch.utils.data import DataLoader


def get_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()

    tot_correct = 0

    for batch in dataloader:
