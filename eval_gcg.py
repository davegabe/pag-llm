import json
from pathlib import Path
from typing import Any
import torch
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from data.data_processor import clean_text
from instantiate import load_model_from_checkpoint
from models.loader import load_model_and_tokenizer
from infer_backward_tinystories import (
    compute_semantic_similarity,
    count_repeated_tokens,
    load_semantic_model,
)
from gcg.gcg_evaluation import GCGResult
from config import CustomLLMPagConfig, apply_config
from models.base_model import BaseLMModel

# CONFIG
GCG_RESULTS = "checkpoints/tinystories-pretokenized-small/gcg_identity.json"
STEPS = 500
INTERVAL = 50
INTERVALS = [INTERVAL*i for i in range(1, STEPS//INTERVAL + 1)]


@torch.no_grad()
def get_batch_perplexity_from_model(model: Any, x_input_ids: torch.Tensor, x_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity batch using a raw HF model (model) that has a .to(device) applied already.
    This mirrors get_batch_perplexity but accepts a bare model (not a lightning wrapper).
    """
    assert x_input_ids.shape == x_attention_mask.shape
    if x_input_ids.ndim == 1:
        x_input_ids = x_input_ids.unsqueeze(0)
        x_attention_mask = x_attention_mask.unsqueeze(0)

    # Infer model device from first parameter
    model_device = next(model.parameters()).device

    if x_input_ids.device != model_device:
        x_input_ids = x_input_ids.to(model_device)
        x_attention_mask = x_attention_mask.to(model_device)

    out = model(x_input_ids, attention_mask=x_attention_mask, return_dict=True)
    out_logits = out.logits

    shift_logits = out_logits[..., :-1, :].contiguous()
    shift_labels = x_input_ids[..., 1:].contiguous()
    shift_attention_mask_batch = x_attention_mask[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    perplexity_batch = torch.exp(
        (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
        / shift_attention_mask_batch.sum(1)
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return perplexity_batch



def mean_sel(tensor: torch.Tensor, idx: torch.Tensor | None = None) -> float:
    if idx is None:
        return float(tensor.mean().item())
    return float(tensor[idx].mean().item())


def compute_naturalness_metrics(
        finished_results: list[GCGResult],
        lightning_model: BaseLMModel,
        ext_model: Any | None = None,
        ext_tokenizer: Any | None = None,
        semantic_model: Any = None,
    ) -> dict:
    """Compute naturalness metrics (prefix PPLs, token duplication, semantic sim) for a list of finished results.

    Returns a dict with keys: mean_attack_prefix_ppl, mean_orig_prefix_ppl, mean_attack_dup, mean_semantic_sim
    """
    # Reformat strings: replace \u2581 with space, normalize whitespace
    attack_texts = [clean_text(r.x_attack_str, lightning_model.tokenizer) for r in finished_results]
    
    # Get original texts
    original_texts = [lightning_model.tokenizer.decode(r.original_prefix_ids, skip_special_tokens=True) for r in finished_results]
    original_texts = [clean_text(t, lightning_model.tokenizer) for t in original_texts]

    if ext_model is None or ext_tokenizer is None:
        raise Exception("External LLM or tokenizer is not available")

    # Tokenize with external tokenizer and move tensors to the evaluated model device
    enc_attack = ext_tokenizer(attack_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    enc_orig = ext_tokenizer(original_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    attack_ids = enc_attack['input_ids']
    attack_mask = enc_attack['attention_mask']
    orig_ids = enc_orig['input_ids']
    orig_mask = enc_orig['attention_mask']

    # compute prefix PPLs using the external model directly
    batch_size = 64
    attack_prefix_ppls = []
    orig_prefix_ppls = []
    for i in range(0, len(attack_ids), batch_size):
        end_i = min(i + batch_size, len(attack_ids))
        batch_attack = attack_ids[i:end_i]
        batch_attack_mask = attack_mask[i:end_i]
        batch_orig = orig_ids[i:end_i]
        batch_orig_mask = orig_mask[i:end_i]
        attack_ppl = get_batch_perplexity_from_model(ext_model, batch_attack, batch_attack_mask)
        orig_ppl = get_batch_perplexity_from_model(ext_model, batch_orig, batch_orig_mask)
        attack_prefix_ppls.append(attack_ppl)
        orig_prefix_ppls.append(orig_ppl)
    attack_prefix_ppls = torch.cat(attack_prefix_ppls)
    orig_prefix_ppls = torch.cat(orig_prefix_ppls)

    mean_attack_prefix_ppl = float(torch.nanmean(attack_prefix_ppls).item()) if attack_prefix_ppls.numel() else float('nan')
    mean_orig_prefix_ppl = float(torch.nanmean(orig_prefix_ppls).item()) if orig_prefix_ppls.numel() else float('nan')

    # Token duplication (repetition)
    attack_dup = count_repeated_tokens(attack_ids, attack_mask).cpu()
    orig_dup = count_repeated_tokens(orig_ids, orig_mask).cpu()
    mean_attack_dup = float(attack_dup.float().mean().item()) if attack_dup.numel() else float('nan')
    mean_orig_dup = float(orig_dup.float().mean().item()) if orig_dup.numel() else float('nan')

    # Semantic similarity
    sims = [compute_semantic_similarity(a, o, semantic_model) for a, o in zip(attack_texts, original_texts)]
    mean_semantic_sim = float(sum(sims) / len(sims)) if sims else float('nan')

    return {
        'mean_attack_prefix_ppl': mean_attack_prefix_ppl,
        'mean_orig_prefix_ppl': mean_orig_prefix_ppl,
        'mean_attack_dup': mean_attack_dup,
        'mean_orig_dup': mean_orig_dup,
        'mean_semantic_sim': mean_semantic_sim,
    }


def compute_step_metrics(
        orig_losses: torch.Tensor | None,
        attack_losses: torch.Tensor | None,
        kl_divs: torch.Tensor | None,
        finished_results: list[GCGResult],
        lightning_model: BaseLMModel,
        total_samples: int,
        ext_model: Any | None = None,
        ext_tokenizer: Any | None = None,
        semantic_model: Any = None,
    ) -> dict:
    """Compute all per-step aggregated metrics and return a dict ready for wandb.log."""
    # Defensive defaults
    if orig_losses is None or attack_losses is None or kl_divs is None:
        return {}

    delta_losses = orig_losses - attack_losses

    success_mask = (attack_losses <= orig_losses)
    sample_success_rate = float(success_mask.float().sum().item() / total_samples)
    successful_samples = int(success_mask.sum().item())

    success_idx = success_mask.nonzero(as_tuple=True)[0]
    fail_idx = (~success_mask).nonzero(as_tuple=True)[0]

    metrics: dict = {}
    # basic counts
    metrics.update({
        'convergence/finished_samples': len(finished_results),
        'convergence/sample_success_rate': sample_success_rate,
        'convergence/successful_samples': successful_samples,
    })

    # loss aggregates
    metrics.update({
        'convergence/orig_loss_mean': mean_sel(orig_losses),
        'convergence/attack_loss_mean': mean_sel(attack_losses),
        'convergence/delta_loss_mean': mean_sel(delta_losses),
    })

    # loss aggregates for success / fail subsets (per-step)
    metrics.update({
        'convergence/orig_loss_mean_success': mean_sel(orig_losses, success_idx),
        'convergence/attack_loss_mean_success': mean_sel(attack_losses, success_idx),
        'convergence/delta_loss_mean_success': mean_sel(delta_losses, success_idx),

        'convergence/orig_loss_mean_fail': mean_sel(orig_losses, fail_idx),
        'convergence/attack_loss_mean_fail': mean_sel(attack_losses, fail_idx),
        'convergence/delta_loss_mean_fail': mean_sel(delta_losses, fail_idx),
    })

    # KL divergence metrics
    metrics.update({
        'convergence/kl_div_mean': mean_sel(kl_divs),
        'convergence/kl_div_mean_success': mean_sel(kl_divs, success_idx),
        'convergence/kl_div_mean_fail': mean_sel(kl_divs, fail_idx),
    })

    # naturalness metrics
    nat = compute_naturalness_metrics(finished_results, lightning_model, ext_model, ext_tokenizer, semantic_model)
    metrics['convergence/attack_prefix_ppl_mean'] = nat['mean_attack_prefix_ppl']
    metrics['convergence/orig_prefix_ppl_mean'] = nat['mean_orig_prefix_ppl']
    metrics['convergence/attack_token_duplications_mean'] = nat['mean_attack_dup']
    metrics['convergence/orig_token_duplications_mean'] = nat['mean_orig_dup']
    metrics['convergence/attack_semantic_similarity_mean'] = nat['mean_semantic_sim']

    # naturalness for success / fail subsets
    def _nat_for_idx(idx: torch.Tensor) -> dict:
        sel = [finished_results[i] for i in idx.tolist()]
        return compute_naturalness_metrics(sel, lightning_model, ext_model, ext_tokenizer, semantic_model)

    nat_succ = _nat_for_idx(success_idx)
    nat_fail = _nat_for_idx(fail_idx)

    metrics['convergence/attack_prefix_ppl_mean_success'] = nat_succ['mean_attack_prefix_ppl']
    metrics['convergence/orig_prefix_ppl_mean_success'] = nat_succ['mean_orig_prefix_ppl']
    metrics['convergence/attack_token_duplications_mean_success'] = nat_succ['mean_attack_dup']
    metrics['convergence/orig_token_duplications_mean_success'] = nat_succ['mean_orig_dup']
    metrics['convergence/attack_semantic_similarity_mean_success'] = nat_succ['mean_semantic_sim']

    metrics['convergence/attack_prefix_ppl_mean_fail'] = nat_fail['mean_attack_prefix_ppl']
    metrics['convergence/orig_prefix_ppl_mean_fail'] = nat_fail['mean_orig_prefix_ppl']
    metrics['convergence/attack_token_duplications_mean_fail'] = nat_fail['mean_attack_dup']
    metrics['convergence/orig_token_duplications_mean_fail'] = nat_fail['mean_orig_dup']
    metrics['convergence/attack_semantic_similarity_mean_fail'] = nat_fail['mean_semantic_sim']

    return metrics


def compute_aggregated_gcg_metrics(
        orig_losses: torch.Tensor | None,
        attack_losses: torch.Tensor | None,
        kl_divs: torch.Tensor | None,
        all_results: list[GCGResult],
        lightning_model: BaseLMModel,
        cfg: CustomLLMPagConfig,
        ext_model: Any | None = None,
        ext_tokenizer: Any | None = None,
        semantic_model: Any = None,
    ) -> dict:
    """Compute run-level aggregated gcg/* metrics for wandb.summary (single values).

    This mirrors the previous gcg/* keys but ensures they're written once as summary values.
    """
    if orig_losses is None or attack_losses is None or kl_divs is None:
        return {}

    delta_losses = orig_losses - attack_losses

    success_mask = (attack_losses <= orig_losses)
    success_idx = success_mask.nonzero(as_tuple=True)[0]
    fail_idx = (~success_mask).nonzero(as_tuple=True)[0]

    nat = compute_naturalness_metrics(all_results, lightning_model, ext_model, ext_tokenizer, semantic_model)

    metrics = {
        'gcg/orig_loss_mean': mean_sel(orig_losses),
        'gcg/attack_loss_mean': mean_sel(attack_losses),
        'gcg/delta_loss_mean': mean_sel(delta_losses),

        'gcg/orig_loss_mean_success': mean_sel(orig_losses, success_idx),
        'gcg/attack_loss_mean_success': mean_sel(attack_losses, success_idx),
        'gcg/delta_loss_mean_success': mean_sel(delta_losses, success_idx),

        'gcg/orig_loss_mean_fail': mean_sel(orig_losses, fail_idx),
        'gcg/attack_loss_mean_fail': mean_sel(attack_losses, fail_idx),
        'gcg/delta_loss_mean_fail': mean_sel(delta_losses, fail_idx),

        'gcg/attack_prefix_ppl_mean': nat['mean_attack_prefix_ppl'],
        'gcg/orig_prefix_ppl_mean': nat['mean_orig_prefix_ppl'],
        'gcg/attack_token_duplications_mean': nat['mean_attack_dup'],
        'gcg/orig_token_duplications_mean': nat['mean_orig_dup'],
        'gcg/attack_semantic_similarity_mean': nat['mean_semantic_sim'],
    }

    # KL divergence metrics
    metrics.update({
        'gcg/kl_div_mean': mean_sel(kl_divs),
        'gcg/kl_div_mean_success': mean_sel(kl_divs, success_idx),
        'gcg/kl_div_mean_fail': mean_sel(kl_divs, fail_idx),
    })

    return metrics


def _compute_loss_metrics(
        gcg_results: list[GCGResult],
        lightning_model: BaseLMModel,
        batch_size: int = 128
    ):
    """Compute per-sample CE loss on the target suffix for both original and attack prefixes.

    Returns tensors: original_losses (N,), attack_losses (N,), kl_divs (N,)
    """
    if not gcg_results:
        return None, None, None

    tokenizer = lightning_model.tokenizer
    llm = lightning_model.model
    device = lightning_model.device

    pad_token_id: Any = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    overall_original_loss = []
    overall_attack_loss = []
    overall_kl = []

    for start_i in range(0, len(gcg_results), batch_size):
        end_i = min(start_i + batch_size, len(gcg_results))
        batch = gcg_results[start_i:end_i]

        # Original prefixes: already token ids
        original_prefix_ids = pad_sequence([torch.tensor(r.original_prefix_ids) for r in batch],
                                           batch_first=True, padding_value=pad_token_id).to(device)
        # Target responses
        target_response_ids = pad_sequence([torch.tensor(r.target_response_ids) for r in batch],
                                           batch_first=True, padding_value=pad_token_id).to(device)

        # Attack prefixes: encode strings with tokenizer (batch encoding handles variable lengths)
        attack_enc = tokenizer([r.x_attack_str for r in batch],
                               return_tensors='pt', padding=True, truncation=True, max_length=512)
        attack_input_ids = attack_enc['input_ids'].to(device)  # type: ignore

        target_len = target_response_ids.size(1)

        # Build target mask for valid positions
        target_lengths = torch.tensor(
            [len(r.target_response_ids) for r in batch], device=device)
        target_mask = (torch.arange(target_len, device=device).unsqueeze(
            0) < target_lengths.unsqueeze(1))

        with torch.no_grad():
            # Compute logits for original prefixes
            original_concat = torch.cat(
                [original_prefix_ids, target_response_ids], dim=1)
            logits_orig = llm(original_concat, return_dict=True).logits
            logits_orig_target = logits_orig[:, -target_len:, :]

            # Compute logits for attack prefixes
            attack_concat = torch.cat(
                [attack_input_ids, target_response_ids], dim=1)
            logits_attack = llm(attack_concat, return_dict=True).logits
            logits_attack_target = logits_attack[:, -target_len:, :]

            # CE loss per sample
            ce_orig = F.cross_entropy(logits_orig_target.transpose(1, 2), target_response_ids,
                                      reduction='none', ignore_index=pad_token_id)
            ce_attack = F.cross_entropy(logits_attack_target.transpose(1, 2), target_response_ids,
                                        reduction='none', ignore_index=pad_token_id)

            masked_orig = ce_orig * target_mask.float()
            masked_attack = ce_attack * target_mask.float()

            orig_loss_per_sample = masked_orig.sum(
                dim=-1) / target_mask.sum(dim=-1).clamp(min=1)
            attack_loss_per_sample = masked_attack.sum(
                dim=-1) / target_mask.sum(dim=-1).clamp(min=1)

            # KL divergence between logits
            vocab_size = logits_orig_target.size(-1)

            kl_div = F.kl_div(
                logits_attack_target.reshape(-1,
                                             vocab_size).log_softmax(dim=-1),
                logits_orig_target.reshape(-1, vocab_size).log_softmax(dim=-1),
                reduction='none', log_target=True
            ).sum(dim=-1).view(len(batch), target_len)
            masked_kl = (kl_div * target_mask.float()).sum(dim=-
                                                           1) / target_mask.sum(dim=-1).clamp(min=1)

            overall_original_loss.append(orig_loss_per_sample.cpu())
            overall_attack_loss.append(attack_loss_per_sample.cpu())
            overall_kl.append(masked_kl.cpu())

    original_losses = torch.cat(overall_original_loss)
    attack_losses = torch.cat(overall_attack_loss)
    kl_divs = torch.cat(overall_kl)
    return original_losses, attack_losses, kl_divs


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    if cfg.model.checkpoint_path is None:
        raise ValueError("Model checkpoint path is not set.")
    lightning_model, _, model_name, cfg = load_model_from_checkpoint(
        cfg.model.checkpoint_path, cfg
    )
    # Force to GPU if available
    if torch.cuda.is_available() and lightning_model.device == torch.device('cpu'):
        lightning_model = lightning_model.to('cuda')
        print("Forced lightning model to GPU")
    lightning_model.eval()
    print(f"Lightning model device: {lightning_model.device}")
    semantic_model = load_semantic_model(cfg)  # Cache SentenceTransformer once

    # Load external local HF model/tokenizer if configured
    ext_model = None
    ext_tokenizer = None
    if getattr(cfg.model, 'local_external_llm_path', None):
        try:
            path_str = str(cfg.model.local_external_llm_path)
            ext_model, ext_tokenizer = load_model_and_tokenizer(path_str, False)
            ext_model.to(lightning_model.device)  # type: ignore
            ext_model.eval()
            print(f"External model loaded and moved to device: {lightning_model.device}")
        except Exception as e:
            print(f"Failed to load or move external LLM: {e}")
            ext_model = None
            ext_tokenizer = None

    # Locate GCG results file based on the loaded config / model name.
    out_dir = Path(str(cfg.model.output_dir))
    results_path = out_dir / f'gcg_{model_name}.json'

    with results_path.open('r') as f:
        print(f"Loading GCG results from {results_path}")
        results = [GCGResult.from_dict(r) for r in json.load(f)]

    # init wandb if you want to log them as a re-run
    name = f"relog_gcg_{model_name}"
    wandb.init(project="pag-llm-gcg-attacks", name=name)

    total_samples = len(results)
    for step in INTERVALS:
        # finished_results = samples that finished by this step
        finished_results = [r for r in results if r.steps <= step]
        if not finished_results:
            print(f"Step {step}: no finished samples.")
            continue

        # Compute loss metrics
        orig_losses, attack_losses, kl_divs = _compute_loss_metrics(
            finished_results, lightning_model, batch_size=256
        )
        if orig_losses is None or attack_losses is None:
            print(f"Step {step}: missing loss metrics.")
            continue
        success_mask = (attack_losses <= orig_losses)
        sample_success_rate = float(
            success_mask.float().sum().item() / total_samples)
        successful_samples = int(success_mask.sum().item())

        # Compute on successful samples naturalness metrics and
        print(f"Step {step}: N_finished={len(finished_results)}, success_rate={sample_success_rate:.4f}, successful={successful_samples}")

        metrics = compute_step_metrics(
            orig_losses,
            attack_losses,
            kl_divs,
            finished_results,
            lightning_model,
            total_samples,
            ext_model=ext_model,
            ext_tokenizer=ext_tokenizer,
            semantic_model=semantic_model,
        )

        # Ensure gcg/* keys are not logged as time-series by filtering them out here.
        metrics_to_log = {k: v for k, v in metrics.items() if not k.startswith('gcg/')}
        wandb.log(metrics_to_log, step=step)

    # Compute run-level aggregated gcg/* metrics and write them once to wandb.summary
    final_gcg_metrics = compute_aggregated_gcg_metrics(
        orig_losses,
        attack_losses,
        kl_divs,
        results,
        lightning_model,
        cfg,
        ext_model=ext_model,
        ext_tokenizer=ext_tokenizer,
        semantic_model=semantic_model,
    )
    for k, v in final_gcg_metrics.items():
        wandb.summary[k] = v

    wandb.finish()


if __name__ == '__main__':
    main() # type: ignore
