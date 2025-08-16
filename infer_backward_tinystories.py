import os.path
import sys
from pathlib import Path

import torch
from evaluate.utils import logging
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from config import CustomLLMPagConfig, apply_config
from inverse_lm_stats import init_evaluation
from models.base_model import BaseLMModel
from models.common import forward_grad_embeddings
from utils.backward_inference_result import BackwardInferenceSampleResult, BackwardInferenceResult


@torch.no_grad()
def backward_infer_prefix(lightning_module: BaseLMModel, use_init: str, reverse_bigram: torch.Tensor | None,
                          suffix_input_ids: torch.Tensor, suffix_attention_mask: torch.Tensor, suffix_length: int,
                          beam_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform backward inference on the given suffix input IDs and attention mask.

    Args:
        lightning_module (BaseLMModel): The model to use for inference.
        use_init (str): The initialization strategy to use ('bigram', 'random', 'pad', 'bos').
        reverse_bigram (torch.Tensor): The reverse bigram tensor. Required for 'bigram' initialization.
        suffix_input_ids (torch.Tensor): The input IDs of the suffix.
        suffix_attention_mask (torch.Tensor): The attention mask of the suffix.
        suffix_length (int): The length of the fixed given suffix.
        beam_size (int): The beam size for the inference.

    Returns:
        torch.Tensor: The updated input IDs with the predicted token at the first position.
        torch.Tensor: The updated attention mask with the first position set to 1.
    """
    assert reverse_bigram is not None or use_init != 'bigram', 'reverse_bigram must be provided for bigram initialization'

    assert suffix_input_ids.shape == suffix_attention_mask.shape
    if suffix_input_ids.ndim == 1:
        # If the input is a single sentence, add a batch dimension
        suffix_input_ids = suffix_input_ids.unsqueeze(0)
        suffix_attention_mask = suffix_attention_mask.unsqueeze(0)

    assert suffix_input_ids.ndim == 2, f'input_ids shape mismatch: {suffix_input_ids.shape} != (batch_size, seq_len)'

    assert suffix_input_ids.size(0) in (1,
                                        beam_size), f'input_ids batch size must be 1 (single sample) or the beam size. Found: {suffix_input_ids.shape[0]}'

    batch_size, seq_len = suffix_input_ids.shape
    vocab_size = lightning_module.tokenizer.vocab_size
    prefix_length = seq_len - suffix_length + 1

    x_input_ids = torch.cat([torch.zeros_like(suffix_input_ids[:, :1]), suffix_input_ids, ], dim=1)
    assert x_input_ids.shape == (batch_size,
                                 seq_len + 1), f'input_ids shape mismatch: {x_input_ids.shape} != ({batch_size}, {seq_len + 1})'

    # Replace the first token, according to the initialization strategy
    # To do that, the input_ids must have a new token for every sentence
    if use_init == 'bigram':
        x_input_ids[:, 0] = reverse_bigram[suffix_input_ids[:, 0]]
    elif use_init == 'random':
        x_input_ids[:, 0] = torch.randint_like(x_input_ids[:, 0], 0, vocab_size)
    elif use_init == 'pad':
        x_input_ids[:, 0] = lightning_module.tokenizer.pad_token_id
    elif use_init == 'bos':
        x_input_ids[:, 0] = lightning_module.tokenizer.bos_token_id
    else:
        raise ValueError(f'Invalid initialization strategy: {use_init}. Allowed values are: bigram, random, pad, bos')

    x_attention_mask = torch.cat([torch.ones_like(suffix_attention_mask[:, :1]), suffix_attention_mask, ], dim=1)
    assert x_attention_mask.shape == (batch_size,
                                      seq_len + 1), f'attention_mask shape mismatch: {x_attention_mask.shape} != ({batch_size}, {seq_len + 1})'

    shift_labels = torch.cat([suffix_input_ids, torch.zeros_like(suffix_input_ids[:, :1]), ], dim=1)
    assert shift_labels.shape == (batch_size,
                                  seq_len + 1), f'shift_labels shape mismatch: {shift_labels.shape} != ({batch_size}, {seq_len + 1})'
    assert shift_labels[:, -1].sum() == 0, f'shift_labels last token must be zero: {shift_labels[:, -1]}'

    # Get the embeddings of X (with the k-th token replaced)
    with torch.set_grad_enabled(True):
        x_embed = lightning_module.model.get_input_embeddings()(x_input_ids).detach()
        x_embed.requires_grad_(True)

        outputs = lightning_module.model(inputs_embeds=x_embed, attention_mask=x_attention_mask, labels='dummy',
                                         shift_labels=shift_labels, )
        grad_x_embed = torch.autograd.grad(outputs.loss, [x_embed], create_graph=False)[0][:, 0]

    # Predict the k-th token, based on the gradients of the first token embeddings
    new_embed = lightning_module.classification_strategy(x_embed[:, 0], grad_x_embed)
    logits = forward_grad_embeddings(lightning_module.model, new_embed)
    assert logits.shape == (batch_size,
                            vocab_size), f'logits shape mismatch: {logits.shape} != ({batch_size}, {vocab_size})'

    # Take the top-K logits, where K is the beam size
    # (token id = index in the vocabulary)
    _, top_k_tokens = torch.topk(logits, beam_size, dim=-1, largest=True, sorted=False)
    assert top_k_tokens.shape == (batch_size,
                                  beam_size), f'top_k_tokens shape mismatch: {top_k_tokens.shape} != ({batch_size}, {beam_size})'
    assert top_k_tokens.dtype == torch.int64, f'top_k_tokens dtype mismatch: {top_k_tokens.dtype} != torch.int64'

    # Now, we must increase the batch size, multiplying it by the beam size
    # and repeat the input_ids and attention_mask
    # Remember that repeat_interleave makes the original tensor [1,2,3] -> [1,1,2,2,3,3]
    x_input_ids = x_input_ids.repeat_interleave(beam_size, dim=0)
    x_attention_mask = x_attention_mask.repeat_interleave(beam_size, dim=0)
    assert x_input_ids.shape == (batch_size * beam_size,
                                 seq_len + 1), f'input_ids shape mismatch: {x_input_ids.shape} != ({batch_size * beam_size}, {seq_len + 1})'
    assert x_attention_mask.shape == (batch_size * beam_size,
                                      seq_len + 1), f'attention_mask shape mismatch: {x_attention_mask.shape} != ({batch_size * beam_size}, {seq_len + 1})'

    # Append to every sentence in the batch, their corresponding top-k tokens
    for sample_i in range(batch_size):
        # We have to set to the portion of x_input_ids that refers to the first sample in the original batch
        # the top-k tokens.
        x_input_ids[sample_i * beam_size:(sample_i + 1) * beam_size, 0] = top_k_tokens[sample_i]

    # For each sample, keep only the top-k sentence with lower perplexity
    x_input_ids, x_attention_mask, perplexities = get_least_perplexity_sentences(lightning_module, x_input_ids,
                                                                                 x_attention_mask, beam_size,
                                                                                 prefix_length)
    assert x_input_ids.shape == (beam_size,
                                 seq_len + 1), f'input_ids shape mismatch: {x_input_ids.shape} != ({beam_size}, {seq_len + 1})'
    assert x_attention_mask.shape == (beam_size,
                                      seq_len + 1), f'attention_mask shape mismatch: {x_attention_mask.shape} != ({beam_size}, {seq_len + 1})'

    # predicted_texts = lightning_module.tokenizer.batch_decode(x_input_ids[:, :10], skip_special_tokens=True)
    # print(f"BEAM:")
    # for i, (perplexity, text) in enumerate(zip(perplexities, predicted_texts)):
    #     print(f"  - {i}: [{perplexity}] {text}")
    # print()

    return x_input_ids, x_attention_mask


@torch.no_grad()
def get_least_perplexity_sentences(lightning_module: BaseLMModel, x_input_ids: torch.Tensor,
                                   x_attention_mask: torch.Tensor, beam_size: int, prefix_length: int) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
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
    assert top_k_indices.shape == (
        beam_size,), f'Expected top_k_indices to be of shape (beam_size,), but got {top_k_indices.shape}'

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
    assert x_input_ids.shape == x_attention_mask.shape, f'input_ids and attention_mask shape mismatch: {x_input_ids.shape} != {x_attention_mask.shape}'
    assert x_input_ids.ndim == 2, f'input_ids shape mismatch: {x_input_ids.shape} != (batch_size, seq_len)'

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
def get_batch_perplexity(lightning_module: BaseLMModel, x_input_ids: torch.Tensor,
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
    assert x_input_ids.shape == x_attention_mask.shape, f'input_ids and attention_mask shape mismatch: {x_input_ids.shape} != {x_attention_mask.shape}'

    if x_input_ids.ndim == 1:
        # If the input is a single sentence, add a batch dimension
        x_input_ids = x_input_ids.unsqueeze(0)
        x_attention_mask = x_attention_mask.unsqueeze(0)
    assert x_input_ids.ndim == 2, f'input_ids shape mismatch: {x_input_ids.shape} != (batch_size, seq_len)'

    out_logits = lightning_module.model(x_input_ids, attention_mask=x_attention_mask).logits

    labels = x_input_ids

    shift_logits = out_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask_batch = x_attention_mask[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss(reduction="none")

    perplexity_batch = torch.exp(
        (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(
            1) / shift_attention_mask_batch.sum(1))
    assert perplexity_batch.shape == (x_input_ids.shape[
                                          0],), f'perplexity_batch shape mismatch: {perplexity_batch.shape} != ({x_input_ids.shape[0]},)'

    return perplexity_batch


def backward_infer_bigram_only(bigram_counts: torch.Tensor | None, lightning_module: BaseLMModel,
                               suffix_input_ids: torch.Tensor, suffix_attention_mask: torch.Tensor,
                               prefix_tokens_len: int, beam_size: int) -> torch.Tensor:
    """
    Perform backward inference using only the reverse bigram tensor.

    Args:
        bigram_counts: The full bigram probabilities tensor, not only the argmax.
        lightning_module: The model to use for inference to compute the perplexity. Not used for prefix prediction.
        suffix_input_ids: Suffix input IDs to be used for inference.
        suffix_attention_mask: Attention mask for the suffix input IDs.
        prefix_tokens_len: The length of the prefix tokens to generate.
        beam_size: The number of beams to use for the search of the prefix with minimum perplexity.

    Returns:
        torch.Tensor: The predicted input IDs after applying the reverse bigram.
    """
    repeated_suffix_input_ids = suffix_input_ids.repeat(beam_size, 1)
    repeated_attn_mask = suffix_attention_mask.repeat(beam_size, 1)
    assert repeated_suffix_input_ids.ndim == 2, f'suffix_input_ids shape mismatch: {repeated_suffix_input_ids.shape} != (batch_size, seq_len)'
    assert repeated_suffix_input_ids.shape == (beam_size,
                                               *suffix_input_ids.shape), f'suffix_input_ids shape mismatch: {repeated_suffix_input_ids.shape} != ({beam_size}, {suffix_input_ids.shape})'

    prefix_tokens = torch.zeros((beam_size, prefix_tokens_len,), dtype=torch.int64, device=suffix_input_ids.device)
    prefix_attn_mask = torch.zeros_like(prefix_tokens, dtype=torch.int64, device=suffix_input_ids.device)

    first_token = repeated_suffix_input_ids[:, 0]  # The first token of the suffix

    for i in range(prefix_tokens_len - 1, -1, -1):
        # Create the batch of possible prefix tokens
        candidate_sentences_ids = torch.cat([prefix_tokens[:, i:], repeated_suffix_input_ids], dim=1).repeat(beam_size,
                                                                                                             1)
        candidate_sentences_attn_mask = torch.cat([prefix_attn_mask[:, i:], repeated_attn_mask], dim=1).repeat(
            beam_size, 1)
        assert candidate_sentences_ids.shape == (beam_size ** 2, prefix_tokens_len - i + suffix_input_ids.shape[
            0]), f'candidate_sentences_ids shape mismatch: {candidate_sentences_ids.shape} != ({beam_size}, {beam_size}, {prefix_tokens_len - i + suffix_input_ids.shape[0]})'
        assert candidate_sentences_attn_mask.shape == candidate_sentences_ids.shape, f'candidate_sentences_attn_mask shape mismatch: {candidate_sentences_attn_mask.shape} != {candidate_sentences_ids.shape}'

        # Use the reverse bigram to get the corresponding prefix token (top-k, not just argmax)
        candidate_sentences_ids[:, 0] = torch.topk(bigram_counts[first_token], k=beam_size, dim=-1).indices.flatten()
        candidate_sentences_attn_mask[:, 0] = 1

        # Choose the sentence with the lowest perplexity / the lowest loss
        prefix_tokens[:, i:] = \
            get_least_perplexity_sentences(lightning_module=lightning_module, x_input_ids=candidate_sentences_ids,
                                           x_attention_mask=candidate_sentences_attn_mask, beam_size=beam_size,
                                           prefix_length=prefix_tokens_len - i, )[
                0][:, i:prefix_tokens_len]
        first_token = prefix_tokens[:, i]

    return prefix_tokens[0]  # Return only the first sentence, which is the best one


def run_evaluation(device: str, prefix_len: int, use_init: str, ckpt_file: str, baseline_ckpt_file: str | None,
                   cfg: CustomLLMPagConfig, k_samples: int | None, skip_prefix_tokens: int, beam_size: int):
    output_file = f'backward_inference-{cfg.training.method}-{use_init}.zip'
    output_file = f'{cfg.model.output_dir}/{output_file}'

    if os.path.exists(output_file):
        print(f'Output file {output_file} already exists. Skipping evaluation.')
        return

    print('Results will be saved to:', output_file)

    lightning_module, _, data_module, reverse_bigram, bigram_counts = init_evaluation(cfg=cfg, device=device,
                                                                                      use_init=use_init,
                                                                                      ckpt_file=ckpt_file, )

    # The baseline model is used in the bigram-only inversion.
    # This is necessary because the bigram-only inversion uses the model to compute the perplexity of the generated sentences.
    # But we have to compare it using the baseline model, not to have the side-effects of the PAG training.
    baseline_model: BaseLMModel | None = None
    if use_init == 'bigram' and baseline_ckpt_file is not None:
        baseline_model, _, _, _, _ = init_evaluation(cfg=cfg, device=device, use_init=use_init,
                                                     ckpt_file=baseline_ckpt_file, )
    elif use_init == 'bigram' and baseline_ckpt_file is None:
        print('[WARNING] No baseline model provided for bigram-only inversion.', file=sys.stderr, flush=True)

    lightning_module.eval()

    # Global sample counter across all batches
    global_sample_idx = 0
    total_samples_processed = 0

    all_samples_processed: list[BackwardInferenceSampleResult] = []

    # Iterate over entire test dataset
    for batch_idx, batch in enumerate(tqdm(data_module.test_dataloader(), desc='Processing inversion batches')):
        batch = batch.to(torch.device(device))
        input_ids, attention_mask, labels, shift_labels = batch
        t = input_ids.size(-1)
        batch_size = input_ids.size(0)

        # Apply k_samples limit across entire dataset, not per batch
        if k_samples is not None:
            remaining_samples = k_samples - total_samples_processed
            if remaining_samples <= 0:
                break
            if remaining_samples < batch_size:
                input_ids = input_ids[:remaining_samples]
                attention_mask = attention_mask[:remaining_samples]
                batch_size = remaining_samples

        # Remove prefix tokens
        input_ids = input_ids[:, skip_prefix_tokens:]
        attention_mask = attention_mask[:, skip_prefix_tokens:]
        t -= skip_prefix_tokens

        assert input_ids.shape == (batch_size, t), f'input_ids shape mismatch: {input_ids.shape} != ({batch_size}, {t})'
        assert attention_mask.shape == (batch_size,
                                        t), f'attention_mask shape mismatch: {attention_mask.shape} != ({batch_size}, {t})'

        # print(f"Processing batch {batch_idx + 1}, samples {global_sample_idx} to {global_sample_idx + batch_size - 1}")

        for local_sample_idx, (sample_input_ids, sample_attention_mask) in enumerate(zip(input_ids, attention_mask)):
            # Give the model the input_ids without the first prefix_len tokens and see what happens
            suffix_input_ids = sample_input_ids[prefix_len:]
            suffix_attention_mask = sample_attention_mask[prefix_len:]
            assert suffix_input_ids.shape == (
                t - prefix_len,), f'suffix_input_ids shape mismatch: {suffix_input_ids.shape} != ({t - prefix_len},)'
            assert suffix_attention_mask.shape == (
                t - prefix_len,), f'suffix_attention_mask shape mismatch: {suffix_attention_mask.shape} != ({t - prefix_len},)'

            suffix_tokens_len = suffix_input_ids.size(-1)
            suffix_text = pretty_decode_tokens(lightning_module.tokenizer, suffix_input_ids)  # Y
            y_tokens = suffix_input_ids.detach().cpu().tolist()  # Y tokens

            # Iteratively replace the first token, in an autoregressive manner
            while suffix_input_ids.size(-1) < t:
                suffix_input_ids, suffix_attention_mask = backward_infer_prefix(lightning_module, use_init,
                                                                                reverse_bigram, suffix_input_ids,
                                                                                suffix_attention_mask,
                                                                                suffix_tokens_len, beam_size, )
            # Since the sentences are sorted, take the first one, which is the best one
            suffix_input_ids, suffix_attention_mask = suffix_input_ids[0], suffix_attention_mask[0]

            prefix_tokens_len = t - suffix_tokens_len

            # Get original prefix text for semantic similarity comparison
            original_prefix_text = pretty_decode_tokens(lightning_module.tokenizer, sample_input_ids)  # X
            predicted_prefix_text = pretty_decode_tokens(lightning_module.tokenizer, suffix_input_ids)  # X'

            bigram_text: str | None = None  # X' bigram-only
            bigram_text_tokens: torch.Tensor | None = None  # X' bigram-only tokens
            if use_init == 'bigram' and baseline_model is not None:
                # Print the generated text using the reverse bigram only
                bigram_text_tokens = backward_infer_bigram_only(bigram_counts, baseline_model, suffix_input_ids,
                                                                suffix_attention_mask, prefix_tokens_len, beam_size, )[
                                     :prefix_tokens_len]
                bigram_text = pretty_decode_tokens(lightning_module.tokenizer, bigram_text_tokens)

            # Finally, save the sample to the output file
            sample_processed = BackwardInferenceSampleResult(
                original_prefix_tokens=sample_input_ids.detach().cpu().tolist(),
                original_prefix_text=original_prefix_text,
                predicted_prefix_tokens=suffix_input_ids.detach().cpu().tolist(),
                predicted_prefix_text=predicted_prefix_text, suffix_tokens=y_tokens, suffix_text=suffix_text,
                bigram_tokens=bigram_text_tokens.detach().cpu().tolist() if bigram_text_tokens is not None else None,
                bigram_text=bigram_text, )
            all_samples_processed.append(sample_processed)

        # Update counters at the end of each batch
        global_sample_idx += batch_size
        total_samples_processed += batch_size

    # At the end of the evaluation, save all samples to the output file
    print(f'Processed {total_samples_processed} samples in total.')
    print(f'Saving results to {output_file}...')
    json_result = BackwardInferenceResult(samples=all_samples_processed, ckpt_file=ckpt_file, prefix_len=prefix_len,
                                          use_init=use_init, baseline_ckpt_file=baseline_ckpt_file,
                                          k_samples=k_samples,
                                          skip_prefix_tokens=skip_prefix_tokens, beam_size=beam_size, )
    json_result.to_file(output_file)


def print_text_stats(lightning_module: BaseLMModel, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     prefix_tokens_len: int, tag: str, ansi_color: str):
    text_ppl = get_batch_perplexity(lightning_module, input_ids, attention_mask).item()

    prefix_ids = input_ids[:prefix_tokens_len]
    prefix_attn_mask = attention_mask[:prefix_tokens_len]
    prefix_text = pretty_decode_tokens(lightning_module.tokenizer, prefix_ids)
    prefix_ppl = get_batch_perplexity(lightning_module, prefix_ids, prefix_attn_mask).item()

    token_duplications = count_repeated_tokens(prefix_ids[None, :], prefix_attn_mask[None, :]).item()

    ansi_colored_text = f"\033[0;{ansi_color}m{prefix_text}\033[0m" if sys.stdout.isatty() else prefix_text

    print(
        f"{tag:<10} [PPL - prefix: {prefix_ppl:>6.1f} / overall: {text_ppl:>6.1f}]: "  # / penalty: {token_duplications}]: "
        f"{ansi_colored_text}")

    # Return metrics for logging
    return {'prefix_ppl': prefix_ppl, 'overall_ppl': text_ppl, 'token_duplications': token_duplications,
            'prefix_text': prefix_text}


def pretty_decode_tokens(tokenizer: PreTrainedTokenizerBase, input_ids: torch.Tensor) -> str:
    """
    Decode the input IDs to a human-readable string, skipping special tokens.
    """
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return decoded_text.replace(' ', '').replace('▁', ' ')


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    logging.disable_progress_bar()

    if "inv-first" in cfg.training.method:
        print(f"Method {cfg.training.method} need to use BOS for initialization, ")
        use_init = 'bos'
    elif "bert-like" in cfg.training.method or "pag-mix-identity-score-embeddings" in cfg.training.method:
        print(f"Method {cfg.training.method} need to use PAD for initialization, ")
        use_init = 'pad'
    elif "identity-grad" in cfg.training.method or cfg.training.method == "base":
        print(f"Method {cfg.training.method} need to use PAG for initialization, ")
        use_init = 'bigram'
    else:
        raise ValueError(f"Unsupported training method: {cfg.training.method}. ")

    ckpt_path = cfg.model.checkpoint_path
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    baseline_ckpt_path = ckpt_path.parent / 'best-base.ckpt'

    run_evaluation(device='cuda:0',
                   k_samples=None,  # How many samples to take from the dataset (set to None for all samples)
                   skip_prefix_tokens=5,  # How many tokens to skip entirely
                   beam_size=5,
                   prefix_len=20,  # How many tokens to predict
                   use_init=use_init,
                   ckpt_file=str(ckpt_path.resolve()),
                   baseline_ckpt_file=str(baseline_ckpt_path.resolve()),
                   cfg=cfg)


if __name__ == '__main__':
    main()
