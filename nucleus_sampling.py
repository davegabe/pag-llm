"""
Implement nucleus sampling for text generation.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def nucleus_sample(logits: torch.Tensor,
                   nucleus_p: float = 0.95,
                   temperature: float = 1.0,
                   beam_size: int = 1) -> torch.Tensor:
    """
    Perform backward inference on the given suffix input IDs and attention mask using Nucleus Sampling.
    This function computes the logits for the next token and selects it using Nucleus Sampling.
    It is a variant of the backward inference that uses Nucleus Sampling instead of beam search.

    Args:
        logits (torch.Tensor): The logits for the next token, shape (batch_size, vocab_size).
        nucleus_p (float): The probability threshold for Nucleus Sampling.
        temperature (float): The temperature to control the randomness of the sampling.
        beam_size (int): The number of beams to use in the sampling. A value > 1 will cause the return tensors to have another dimension of size `beam_size`.

    Returns:
        torch.Tensor: The updated input IDs with the predicted token at the first position.
        torch.Tensor: The updated attention mask with the first position set to 1.
    """
    assert 0.1 <= nucleus_p < 1.0, \
        f'nucleus_p must be in [0.1, 1.0), but got {nucleus_p}'

    batch_size, vocab_size = logits.shape

    # Apply a temperature to the logits to control the randomness of the sampling
    logits = logits / temperature

    # Use Nucleus Sampling to select the next token
    # Source: https://arxiv.org/pdf/1904.09751

    # Compute the probabilities from the logits
    probabilities = F.softmax(logits, dim=-1)

    # Sort the probabilities and their indices, to find the cumulative probabilities later
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)

    # Compute the cumulative probabilities to find the nucleus limit
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the indices of the tokens that are below the nucleus probability threshold
    mask = cumulative_probs <= nucleus_p
    # Shift the mask to include the last token that exceeds the threshold
    mask = torch.cat([
        torch.zeros_like(mask[:, :1]),  # Keep the first token always
        mask[:, :-1],
    ], dim=1)

    # Zero out the probabilities of the tokens that are not in the nucleus
    sorted_probs[mask] = 0.0

    # Rescale the new probabilities
    nucleus_p_sum = sorted_probs.sum(dim=-1, keepdim=True)
    new_probs = sorted_probs / nucleus_p_sum

    # Restore the original indices of the tokens
    probabilities.scatter_(1, sorted_indices, new_probs)

    # Finally, we can sample the next token from the probabilities
    next_tokens = torch.multinomial(probabilities, num_samples=beam_size)
    if beam_size == 1:
        next_tokens = next_tokens.squeeze(1)
        assert next_tokens.shape == (batch_size,), \
            f'next_tokens shape mismatch: {next_tokens.shape} != ({batch_size},)'
    else:
        assert next_tokens.shape == (batch_size, beam_size), \
            f'next_tokens shape mismatch: {next_tokens.shape} != ({batch_size}, {beam_size})'

    return next_tokens
