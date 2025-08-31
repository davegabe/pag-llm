import json
from pathlib import Path
from typing import Any
import torch
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from instantiate import load_model_from_checkpoint
from gcg.gcg_evaluation import GCGResult
from config import CustomLLMPagConfig, apply_config
from models.base_model import BaseLMModel

# CONFIG
GCG_RESULTS = "checkpoints/tinystories-pretokenized-small/gcg_identity.json"
STEPS = 500
INTERVAL = 50
INTERVALS = [INTERVAL*i for i in range(1, STEPS//INTERVAL + 1)]


def _compute_loss_metrics(gcg_results: list[GCGResult], lightning_model: BaseLMModel, batch_size: int = 128):
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
                               return_tensors='pt', padding=True, truncation=True)
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
    lightning_model.eval()

    # Load results
    with Path(GCG_RESULTS).open('r') as f:
        results = [GCGResult.from_dict(r) for r in json.load(f)]

    # init wandb if you want to log them as a re-run
    name = f"relog_{GCG_RESULTS.split('/')[-1].replace('.json','')}"
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
            finished_results, lightning_model, batch_size=128
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
        # Log to wandb: use fixed metric names and pass `step` so WandB plots steps on the X axis
        wandb.log({
            "convergence/finished_samples": len(finished_results),
            "convergence/sample_success_rate": sample_success_rate,
            "convergence/successful_samples": successful_samples,
        }, step=step)


    wandb.finish()


if __name__ == '__main__':
    main() # type: ignore
