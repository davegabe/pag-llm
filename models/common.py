import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def forward_grad_embeddings(model: PreTrainedModel, grad_x_embed: torch.Tensor) -> torch.Tensor:
    """
    Project the gradients of the embeddings to the vocabulary space using the head of the model.

    Args:
        model (PreTrainedModel): The pre-trained model to fine-tune.
        grad_x_embed (torch.Tensor): The gradients of the embeddings. [batch_size, seq_len, embed_dim]

    Returns:
        logits (torch.Tensor): The logits of the model. [batch_size, seq_len, vocab_size]
    """
    # Apply the model normalization to the gradients
    grad_x_embed = model.model.norm(grad_x_embed)  # [batch_size, embed_dim]

    # Create copy of lm_head weights to avoid affecting existing gradients
    lm_head_weight = model.lm_head.weight.clone().detach()

    if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
        lm_head_bias = model.lm_head.bias.clone().detach()
    else:
        lm_head_bias = None

    # Manually compute the projection using copied weights
    logits = F.linear(grad_x_embed, lm_head_weight, lm_head_bias)

    return logits


def compute_top_k_accuracies(inv_first_label: torch.Tensor, logits: torch.Tensor, k_samples: int, tag: str = 'val') -> \
dict[str, torch.Tensor]:
    assert (inv_first_label is None) == (logits is None), \
        "Either both inv_first_label and logits should be None or neither should be None."

    accuracies: dict[str, torch.Tensor] = dict()

    if inv_first_label is None or logits is None:
        for k in range(k_samples):
            accuracies[f'{tag}/{k + 1}_acc'] = torch.tensor(0.0)
            accuracies[f'{tag}/top_{k + 1}_acc'] = torch.tensor(0.0)
        return accuracies


    # Get the probabilities of the model
    probs = F.softmax(logits, dim=-1)  # [batch_size, vocab_size]

    # Get the top k indices
    _, top_k_indices = torch.topk(probs, k_samples, dim=-1, largest=True, sorted=False)  # [batch_size, k]

    # Get the batch size
    n = logits.size(0)

    # Check if the first token is in the k-nearest neighbors
    is_in_k_nearest = torch.zeros((n, k_samples), device=logits.device, dtype=torch.bool)  # [batch_size, k]
    for k in range(k_samples):
        is_in_k_nearest[:, k] = (inv_first_label == top_k_indices[:, k])

    # Calculate the accuracy on the k-nearest neighbors
    for k in range(k_samples):
        # Log the accuracy at exact k position
        acc = is_in_k_nearest[:, k].float().mean()
        accuracies[f'{tag}/{k + 1}_acc'] = acc

        # Log the accuracy for the first k positions
        acc = is_in_k_nearest[:, :k + 1].any(dim=1).float().mean()
        accuracies[f'{tag}/top_{k + 1}_acc'] = acc

    return accuracies
