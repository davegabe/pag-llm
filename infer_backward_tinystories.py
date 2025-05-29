import torch
from torch.nn import CrossEntropyLoss

import nucleus_sampling
from config import CustomLLMPagConfig, apply_config
from data.data_processor import BatchType
from inverse_lm_stats import init_evaluation
from models.base_model import BaseLMModel
from models.common import forward_grad_embeddings


@torch.no_grad()
def backward_infer_prefix_beam_search(lightning_module: BaseLMModel,
                                      use_init: str,
                                      reverse_bigram: torch.Tensor | None,
                                      suffix_input_ids: torch.Tensor,
                                      suffix_attention_mask: torch.Tensor,
                                      suffix_length: int,
                                      beam_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform backward inference on the given suffix input IDs and attention mask.

    Args:
        lightning_module (BaseLMModel): The model to use for inference.
        use_init (str): The initialization strategy to use ('bigram', 'random', or 'pad').
        reverse_bigram (torch.Tensor): The reverse bigram tensor. Required for 'bigram' initialization.
        suffix_input_ids (torch.Tensor): The input IDs of the suffix.
        suffix_attention_mask (torch.Tensor): The attention mask of the suffix.
        suffix_length (int): The length of the fixed given suffix.
        beam_size (int): The beam size for the inference.

    Returns:
        torch.Tensor: The updated input IDs with the predicted token at the first position.
        torch.Tensor: The updated attention mask with the first position set to 1.
    """
    logits, x_input_ids, x_attention_mask = _backward_infer_prefix_logits(
        lightning_module=lightning_module,
        use_init=use_init,
        reverse_bigram=reverse_bigram,
        suffix_input_ids=suffix_input_ids,
        suffix_attention_mask=suffix_attention_mask,
        beam_size=beam_size,
    )
    batch_size, vocab_size = logits.shape
    seq_len = suffix_input_ids.size(-1)
    prefix_length = seq_len - suffix_length + 1

    # Take the top-K logits, where K is the beam size
    # (token id = index in the vocabulary)
    _, top_k_tokens = torch.topk(logits, beam_size, dim=-1, largest=True, sorted=False)
    assert top_k_tokens.shape == (batch_size, beam_size), \
        f'top_k_tokens shape mismatch: {top_k_tokens.shape} != ({batch_size}, {beam_size})'
    assert top_k_tokens.dtype == torch.int64, \
        f'top_k_tokens dtype mismatch: {top_k_tokens.dtype} != torch.int64'

    # Now, we must increase the batch size, multiplying it by the beam size
    # and repeat the input_ids and attention_mask
    # Remember that repeat_interleave makes the original tensor [1,2,3] -> [1,1,2,2,3,3]
    x_input_ids = x_input_ids.repeat_interleave(beam_size, dim=0)
    x_attention_mask = x_attention_mask.repeat_interleave(beam_size, dim=0)
    assert x_input_ids.shape == (batch_size * beam_size, seq_len + 1), \
        f'input_ids shape mismatch: {x_input_ids.shape} != ({batch_size * beam_size}, {seq_len + 1})'
    assert x_attention_mask.shape == (batch_size * beam_size, seq_len + 1), \
        f'attention_mask shape mismatch: {x_attention_mask.shape} != ({batch_size * beam_size}, {seq_len + 1})'

    # Append to every sentence in the batch, their corresponding top-k tokens
    for sample_i in range(batch_size):
        # We have to set to the portion of x_input_ids that refers to the first sample in the original batch
        # the top-k tokens.
        x_input_ids[sample_i * beam_size:(sample_i + 1) * beam_size, 0] = top_k_tokens[sample_i]

    # For each sample, keep only the top-k sentence with lower perplexity
    x_input_ids, x_attention_mask, perplexities = get_least_perplexity_sentences(lightning_module, x_input_ids,
                                                                                 x_attention_mask, beam_size,
                                                                                 prefix_length)
    assert x_input_ids.shape == (beam_size, seq_len + 1), \
        f'input_ids shape mismatch: {x_input_ids.shape} != ({beam_size}, {seq_len + 1})'
    assert x_attention_mask.shape == (beam_size, seq_len + 1), \
        f'attention_mask shape mismatch: {x_attention_mask.shape} != ({beam_size}, {seq_len + 1})'

    # predicted_texts = lightning_module.tokenizer.batch_decode(x_input_ids[:, :10], skip_special_tokens=True)
    # print(f"BEAM:")
    # for i, (perplexity, text) in enumerate(zip(perplexities, predicted_texts)):
    #     print(f"  - {i}: [{perplexity}] {text}")
    # print()

    return x_input_ids, x_attention_mask


@torch.no_grad()
def _backward_infer_prefix_logits(lightning_module: BaseLMModel,
                                  use_init: str,
                                  reverse_bigram: torch.Tensor | None,
                                  suffix_input_ids: torch.Tensor,
                                  suffix_attention_mask: torch.Tensor,
                                  beam_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform backward inference on the given suffix input IDs and attention mask.

    Args:
        lightning_module (BaseLMModel): The model to use for inference.
        use_init (str): The initialization strategy to use ('bigram', 'random', or 'pad').
        reverse_bigram (torch.Tensor): The reverse bigram tensor. Required for 'bigram' initialization.
        suffix_input_ids (torch.Tensor): The input IDs of the suffix.
        suffix_attention_mask (torch.Tensor): The attention mask of the suffix.
        beam_size (int): The beam size for the inference.

    Returns:
        torch.Tensor: logits for the predicted token at the first position.
        torch.Tensor: The updated input IDs with the init token at the first position.
        torch.Tensor: The updated attention mask with the first position set to 1.
    """
    assert reverse_bigram is not None or use_init != 'bigram', \
        'reverse_bigram must be provided for bigram initialization'

    assert suffix_input_ids.shape == suffix_attention_mask.shape
    if suffix_input_ids.ndim == 1:
        # If the input is a single sentence, add a batch dimension
        suffix_input_ids = suffix_input_ids.unsqueeze(0)
        suffix_attention_mask = suffix_attention_mask.unsqueeze(0)

    assert suffix_input_ids.ndim == 2, \
        f'input_ids shape mismatch: {suffix_input_ids.shape} != (batch_size, seq_len)'

    assert suffix_input_ids.size(0) in (1, beam_size), \
        f'input_ids batch size must be 1 (single sample) or beam size (= {beam_size}). Found: {suffix_input_ids.shape[0]}'

    batch_size, seq_len = suffix_input_ids.shape
    vocab_size = lightning_module.tokenizer.vocab_size

    x_input_ids = torch.cat([
        torch.zeros_like(suffix_input_ids[:, :1]),
        suffix_input_ids,
    ], dim=1)
    assert x_input_ids.shape == (batch_size, seq_len + 1), \
        f'input_ids shape mismatch: {x_input_ids.shape} != ({batch_size}, {seq_len + 1})'

    # Replace the first token, according to the initialization strategy
    # To do that, the input_ids must have a new token for every sentence
    if use_init == 'bigram':
        x_input_ids[:, 0] = reverse_bigram[suffix_input_ids[:, 0]]
    elif use_init == 'random':
        x_input_ids[:, 0] = torch.randint_like(x_input_ids[:, 0], 0, vocab_size)
    elif use_init == 'pad':
        x_input_ids[:, 0] = lightning_module.tokenizer.pad_token_id
    else:
        raise ValueError(f'Invalid initialization strategy: {use_init}. Allowed values are: bigram, random, pad')

    x_attention_mask = torch.cat([
        torch.ones_like(suffix_attention_mask[:, :1]),
        suffix_attention_mask,
    ], dim=1)
    assert x_attention_mask.shape == (batch_size, seq_len + 1), \
        f'attention_mask shape mismatch: {x_attention_mask.shape} != ({batch_size}, {seq_len + 1})'

    shift_labels = torch.cat([
        suffix_input_ids,
        torch.zeros_like(suffix_input_ids[:, :1]),
    ], dim=1)
    assert shift_labels.shape == (batch_size, seq_len + 1), \
        f'shift_labels shape mismatch: {shift_labels.shape} != ({batch_size}, {seq_len + 1})'
    assert shift_labels[:, -1].sum() == 0, \
        f'shift_labels last token must be zero: {shift_labels[:, -1]}'

    # Get the embeddings of X (with the k-th token replaced)
    with torch.set_grad_enabled(True):
        x_embed = lightning_module.model.get_input_embeddings()(x_input_ids).detach()
        x_embed.requires_grad_(True)

        outputs = lightning_module.model(
            inputs_embeds=x_embed,
            attention_mask=x_attention_mask,
            labels='dummy',
            shift_labels=shift_labels,
        )
        grad_x_embed = torch.autograd.grad(outputs.loss, [x_embed], create_graph=False)[0][:, 0]

    # Predict the k-th token, based on the gradients of the first token embeddings
    logits = forward_grad_embeddings(lightning_module.model, grad_x_embed)
    assert logits.shape == (batch_size, vocab_size), \
        f'logits shape mismatch: {logits.shape} != ({batch_size}, {vocab_size})'

    return logits, x_input_ids, x_attention_mask


@torch.no_grad()
def backward_infer_prefix_nucleus_sampling(lightning_module: BaseLMModel,
                                           use_init: str,
                                           reverse_bigram: torch.Tensor | None,
                                           suffix_input_ids: torch.Tensor,
                                           suffix_attention_mask: torch.Tensor,
                                           nucleus_p: float = 0.95,
                                           temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform backward inference on the given suffix input IDs and attention mask using Nucleus Sampling.
    This function computes the logits for the next token and selects it using Nucleus Sampling.
    It is a variant of the backward inference that uses Nucleus Sampling instead of beam search.

    Args:
        lightning_module (BaseLMModel): The model to use for inference.
        use_init (str): The initialization strategy to use ('bigram', 'random', or 'pad').
        reverse_bigram (torch.Tensor | None): The reverse bigram tensor. Required for 'bigram' initialization.
        suffix_input_ids (torch.Tensor): The input IDs of the suffix.
        suffix_attention_mask (torch.Tensor): The attention mask of the suffix.
        nucleus_p (float): The probability threshold for Nucleus Sampling.
        temperature (float): The temperature to control the randomness of the sampling.

    Returns:
        torch.Tensor: The updated input IDs with the predicted token at the first position.
        torch.Tensor: The updated attention mask with the first position set to 1.
    """
    assert 0.1 <= nucleus_p < 1.0, \
        f'nucleus_p must be in [0.1, 1.0), but got {nucleus_p}'

    logits, x_input_ids, x_attention_mask = _backward_infer_prefix_logits(
        lightning_module=lightning_module,
        use_init=use_init,
        reverse_bigram=reverse_bigram,
        suffix_input_ids=suffix_input_ids,
        suffix_attention_mask=suffix_attention_mask,
        beam_size=1,
    )

    next_tokens = nucleus_sampling.nucleus_sample(logits, nucleus_p, temperature)

    x_input_ids[:, 0] = next_tokens
    return x_input_ids, x_attention_mask


@torch.no_grad()
def get_least_perplexity_sentences(lightning_module: BaseLMModel,
                                   x_input_ids: torch.Tensor,
                                   x_attention_mask: torch.Tensor,
                                   beam_size: int,
                                   prefix_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the least perplexity sentences from the given input IDs and attention mask.
    This function computes the perplexity of each sentence and returns the top-k sentences with the lowest perplexity.

    Args:
        lightning_module: The model to use for inference.
        x_input_ids: Input IDs of the sentences.
        x_attention_mask: Attention mask of the sentences.
        beam_size: Beam size for the inference, which corresponds to the number of sentences to return.
        prefix_length: The length of the prefix to consider for token duplication calculation.

    Returns:
        tuple: A tuple containing the top-k input IDs, attention mask, and their corresponding perplexities.
    """
    perplexity_batch = get_batch_perplexity(lightning_module, x_input_ids, x_attention_mask)
    # Penalize repeated tokens in x_input_ids, being aware of the attention mask
    token_repetitions = count_repeated_tokens(x_input_ids[:, :prefix_length], x_attention_mask[:, :prefix_length])
    perplexity_batch += token_repetitions

    # Get the top-k sentences with the lowest perplexity
    _, top_k_indices = torch.topk(perplexity_batch, beam_size, dim=0, largest=False, sorted=True)
    assert top_k_indices.shape == (beam_size,), \
        f'Expected top_k_indices to be of shape (beam_size,), but got {top_k_indices.shape}'

    return x_input_ids[top_k_indices], x_attention_mask[top_k_indices], perplexity_batch[top_k_indices]


@torch.no_grad()
def count_repeated_tokens(x_input_ids: torch.Tensor, x_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Count the number of repeated tokens in the given input IDs, considering the attention mask for valid tokens.
    This function computes the number of repeated tokens for each sentence in the batch.

    Args:
        x_input_ids: Input IDs of the sentences.
        x_attention_mask: Attention mask of the sentences.

    Returns:
        torch.Tensor: The number of repeated tokens for each sentence.
    """
    assert x_input_ids.shape == x_attention_mask.shape, \
        f'input_ids and attention_mask shape mismatch: {x_input_ids.shape} != {x_attention_mask.shape}'
    assert x_input_ids.ndim == 2, \
        f'input_ids shape mismatch: {x_input_ids.shape} != (batch_size, seq_len)'

    num_classes = x_input_ids.max().item() + 1

    # Set an invalid token ID where the attention_mask is zero
    invalid_token_id = num_classes
    x_input_ids = x_input_ids.masked_fill(x_attention_mask == 0, invalid_token_id)

    # Get the counts of each token in the batch, keeping the batch dimension
    # From: https://discuss.pytorch.org/t/batched-bincount/72819/4
    target = torch.zeros(x_input_ids.size(0), invalid_token_id + 1, dtype=x_input_ids.dtype, device=x_input_ids.device)
    values = torch.ones_like(x_input_ids)
    target.scatter_add_(dim=1, index=x_input_ids, src=values)

    # Remove the invalid token id from the target
    target = target[:, :invalid_token_id]

    # Now, remove zeros (tokens not showing in the input_ids) and ones (tokens not repeating) from the target
    target = target.masked_fill_(target < 2, 1)
    return (target - 1).sum(dim=1)


@torch.no_grad()
def get_batch_perplexity(lightning_module: BaseLMModel,
                         x_input_ids: torch.Tensor,
                         x_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the perplexity of the given input IDs and attention mask.
    This function computes the perplexity of each sentence in the batch and returns a tensor containing the perplexity.

    The following code is taken from:
    https://github.com/huggingface/evaluate/blob/5aa3982a9a8c86/metrics/perplexity/perplexity.py#L179

    Args:
        lightning_module: The model to use for inference.
        x_input_ids: Input IDs of the sentences.
        x_attention_mask: Attention mask of the sentences.

    Returns:
        torch.Tensor: The perplexity of the sentences.
    """
    assert x_input_ids.shape == x_attention_mask.shape, \
        f'input_ids and attention_mask shape mismatch: {x_input_ids.shape} != {x_attention_mask.shape}'

    if x_input_ids.ndim == 1:
        # If the input is a single sentence, add a batch dimension
        x_input_ids = x_input_ids.unsqueeze(0)
        x_attention_mask = x_attention_mask.unsqueeze(0)
    assert x_input_ids.ndim == 2, \
        f'input_ids shape mismatch: {x_input_ids.shape} != (batch_size, seq_len)'

    out_logits = lightning_module.model(x_input_ids, attention_mask=x_attention_mask).logits

    labels = x_input_ids

    shift_logits = out_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask_batch = x_attention_mask[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss(reduction="none")

    perplexity_batch = torch.exp(
        (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
        / shift_attention_mask_batch.sum(1)
    )
    assert perplexity_batch.shape == (x_input_ids.shape[0],), \
        f'perplexity_batch shape mismatch: {perplexity_batch.shape} != ({x_input_ids.shape[0]},)'

    return perplexity_batch


def run_evaluation(device: str, prefix_len: int, use_init: str, ckpt_file: str, cfg: CustomLLMPagConfig,
                   k_samples: int, skip_prefix_tokens: int, beam_size: int):
    lightning_module, _, data_module, reverse_bigram, bigram_counts = init_evaluation(
        cfg=cfg,
        device=device,
        use_init=use_init,
        ckpt_file=ckpt_file,
    )

    lightning_module.eval()

    batch: BatchType = next(iter(data_module.val_dataloader(shuffle=True))).to(torch.device(device))

    input_ids, attention_mask, shift_labels = batch.input_ids, batch.attention_mask, batch.shift_labels
    t = input_ids.size(-1)

    # Take only a few samples
    input_ids, attention_mask = input_ids[:k_samples], attention_mask[:k_samples]
    assert input_ids.shape == (k_samples, t), \
        f'input_ids shape mismatch: {input_ids.shape} != ({k_samples}, {t})'
    assert attention_mask.shape == (k_samples, t), \
        f'attention_mask shape mismatch: {attention_mask.shape} != ({k_samples}, {t})'

    # And remove a few prefix tokens
    input_ids, attention_mask = input_ids[:, skip_prefix_tokens:], attention_mask[:, skip_prefix_tokens:]
    t -= skip_prefix_tokens
    assert input_ids.shape == (k_samples, t), \
        f'input_ids shape mismatch: {input_ids.shape} != ({k_samples}, {t})'
    assert attention_mask.shape == (k_samples, t), \
        f'attention_mask shape mismatch: {attention_mask.shape} != ({k_samples}, {t})'

    for sample_input_ids, sample_attention_mask in zip(input_ids, attention_mask):
        # Give the model the input_ids without the first prefix_len tokens and see what happens
        suffix_input_ids = sample_input_ids[prefix_len:]
        suffix_attention_mask = sample_attention_mask[prefix_len:]
        assert suffix_input_ids.shape == (t - prefix_len,), \
            f'suffix_input_ids shape mismatch: {suffix_input_ids.shape} != ({t - prefix_len},)'
        assert suffix_attention_mask.shape == (t - prefix_len,), \
            f'suffix_attention_mask shape mismatch: {suffix_attention_mask.shape} != ({t - prefix_len},)'

        suffix_tokens_len = suffix_input_ids.size(-1)
        suffix_text = lightning_module.tokenizer.decode(suffix_input_ids, skip_special_tokens=True)

        # Iteratively replace the first token, in an autoregressive manner
        while suffix_input_ids.size(-1) < t:
            suffix_input_ids, suffix_attention_mask = backward_infer_prefix_nucleus_sampling(
                lightning_module,
                use_init,
                reverse_bigram,
                suffix_input_ids,
                suffix_attention_mask,
                nucleus_p=0.8,
                temperature=0.8,
            )
        # Since the sentences are sorted, take the first one, which is the best one
        suffix_input_ids, suffix_attention_mask = suffix_input_ids[0], suffix_attention_mask[0]

        prefix_tokens_len = t - suffix_tokens_len

        # Finally, print the predicted sentence
        print_text_stats(
            lightning_module=lightning_module,
            input_ids=suffix_input_ids,
            attention_mask=suffix_attention_mask,
            prefix_tokens_len=prefix_tokens_len,
            tag='Predicted',
            ansi_color='36',
        )

        print_text_stats(
            lightning_module=lightning_module,
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask,
            prefix_tokens_len=prefix_tokens_len,
            tag='Original',
            ansi_color='32',
        )

        print(f"\033[1m[...]\033[0m {suffix_text}")

        print()


def print_text_stats(lightning_module: BaseLMModel, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     prefix_tokens_len: int, tag: str, ansi_color: str):
    text_ppl = get_batch_perplexity(lightning_module, input_ids, attention_mask).item()

    prefix_ids = input_ids[:prefix_tokens_len]
    prefix_attn_mask = attention_mask[:prefix_tokens_len]
    prefix_text = lightning_module.tokenizer.decode(prefix_ids, skip_special_tokens=True)
    prefix_ppl = get_batch_perplexity(lightning_module, prefix_ids, prefix_attn_mask).item()

    token_duplications = count_repeated_tokens(prefix_ids[None, :], prefix_attn_mask[None, :]).item()

    print(f"{tag} [PPL - prefix: {prefix_ppl:.4f} / overall: {text_ppl:.4f} / penalty: {token_duplications}]: "
          f"\033[0;{ansi_color}m{prefix_text}\033[0m")


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig):
    run_evaluation(device='cuda:0',
                   k_samples=30,  # How many samples to take from the dataset
                   skip_prefix_tokens=5,  # How many tokens to skip entirely
                   beam_size=20,
                   prefix_len=20,  # How many tokens to predict
                   use_init='bigram',  # How to initialize the first token
                   ckpt_file='tinystories_identity_grad_norm__qp6q1mop.ckpt',
                   cfg=cfg)


if __name__ == '__main__':
    main()
