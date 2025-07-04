import dataclasses
import json
import pathlib
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import wandb

from config import CustomLLMPagConfig, apply_config
from data.data_processor import TextDataset
from gcg import gcg_algorithm, gcg_evaluation
from instantiate import load_model_from_checkpoint
from models.base_model import BaseLMModel


def run_gcg_single_attack(gcg: gcg_algorithm.GCG, target_response: str):
    start_time = time.time()
    x_attack_str, y_attack_response, attack_stats = gcg.run(target_response,
                                                           evaluate_every_n_steps=50,
                                                           stop_after_same_loss_steps=10)
    attack_time = time.time() - start_time
    
    print(f"Attack string: {x_attack_str}")
    print(f"Attack response: {y_attack_response}")
    print(f"Desired response: {target_response}")
    
    # Log single attack results to wandb
    wandb.log({
        "single_attack/attack_time": attack_time,
        "single_attack/target_response": target_response,
        "single_attack/attack_string": x_attack_str,
        "single_attack/attack_response": y_attack_response,
        "single_attack/success": target_response in y_attack_response
    })


def run_full_gcg_evaluation(gcg: gcg_algorithm.GCG, dataset: TextDataset, gcg_output_file: pathlib.Path):
    print('Attacking:', gcg_output_file.stem, 'on', gcg.device)
    
    # Log GCG configuration
    wandb.log({
        "gcg_config/num_prefix_tokens": gcg.num_prefix_tokens,
        "gcg_config/num_steps": gcg.num_steps,
        "gcg_config/search_width": gcg.search_width,
        "gcg_config/top_k": gcg.top_k,
        "gcg_config/device": str(gcg.device),
        "gcg_config/dataset_size": len(dataset)
    })
    
    start_time = time.time()
    # Set random seed for reproducibility
    torch.manual_seed(42)
    gcg_results = gcg_evaluation.evaluate_model_with_gcg(gcg, dataset,
                                                         target_response_len=10,
                                                         max_samples_to_attack=2_000,
                                                         random_select_samples=True)
    evaluation_time = time.time() - start_time
    
    # Log evaluation time and basic stats
    wandb.log({
        "evaluation/total_time": evaluation_time,
        "evaluation/samples_attacked": len(gcg_results),
        "evaluation/avg_time_per_sample": evaluation_time / len(gcg_results) if gcg_results else 0
    })
    
    with gcg_output_file.open('w') as f:
        json.dump([r.to_dict() for r in gcg_results], f, indent=4)
    print(f"Saved GCG results to {gcg_output_file}")


def compute_success_rate(gcg_results: list[gcg_evaluation.GCGResult]) -> float:
    # Count the number of successfully attacked tokens
    num_successful_tokens = sum(
        result.get_success_tokens()
        for result in gcg_results
    )
    num_total_tokens = sum(
        min(len(result.target_response_ids), len(result.y_attack_response_ids))
        for result in gcg_results
    )
    token_attack_success_rate = num_successful_tokens / num_total_tokens if num_total_tokens > 0 else 0

    # Log token-level success metrics
    wandb.log({
        "success_metrics/token_success_rate": token_attack_success_rate,
        "success_metrics/successful_tokens": num_successful_tokens,
        "success_metrics/total_tokens": num_total_tokens
    })

    return token_attack_success_rate


def compute_mean_steps_to_success(gcg_results: list[gcg_evaluation.GCGResult]) -> tuple[float, float]:
    # Count the average required steps to converge
    # Ignore attacks with less than 2 successful tokens
    successful_steps = [
        result.steps
        for result in gcg_results
    ]
    mean_success_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0
    stddev_success_steps = (sum((x - mean_success_steps) ** 2 for x in successful_steps) / len(
        successful_steps)) ** 0.5 if successful_steps else 0

    # Log step statistics
    wandb.log({
        "convergence/mean_steps_to_success": mean_success_steps,
        "convergence/stddev_steps_to_success": stddev_success_steps,
        "convergence/num_successful_attacks": len(successful_steps)
    })

    return mean_success_steps, stddev_success_steps


def compute_attack_losses(gcg_results: list[gcg_evaluation.GCGResult], lightning_model: BaseLMModel,
                          batch_size: int = 128) -> tuple:
    # Compute the KL-divergence and CE-loss of the target suffix, given both the attack and the original prefix
    tokenizer, llm = lightning_model.tokenizer, lightning_model.model
    overall_original_loss, overall_attack_loss, overall_kl_div = [], [], []
    for start_i in range(0, len(gcg_results), batch_size):
        end_i = min(start_i + batch_size, len(gcg_results))
        batch = gcg_results[start_i:end_i]
        
        # Handle tokenizers that might not have a pad token
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        original_prefix_ids = pad_sequence([torch.tensor(result.original_prefix_ids) for result in batch], 
                                         batch_first=True, padding_value=pad_token_id).to(llm.device)
        target_response_ids = pad_sequence([torch.tensor(result.target_response_ids) for result in batch], 
                                         batch_first=True, padding_value=pad_token_id).to(llm.device)
        y_attack_response_ids = pad_sequence([torch.tensor(result.y_attack_response_ids) for result in batch], 
                                           batch_first=True, padding_value=pad_token_id).to(llm.device)
        x_attack_ids = tokenizer.batch_encode_plus([result.x_attack_str for result in batch],
                                                   return_tensors='pt', padding=True)['input_ids'].to(llm.device)

        prefix_len = original_prefix_ids.size(1)
        batch_size_actual, max_suffix_len = y_attack_response_ids.shape
        vocab_size = tokenizer.vocab_size
        
        # Create attention masks for variable length sequences
        target_lengths = torch.tensor([len(result.target_response_ids) for result in batch], device=llm.device)
        attack_lengths = torch.tensor([len(result.y_attack_response_ids) for result in batch], device=llm.device)
        
        # Create masks for valid positions (non-padded)
        target_mask = torch.arange(max_suffix_len, device=llm.device).unsqueeze(0) < target_lengths.unsqueeze(1)
        attack_mask = torch.arange(max_suffix_len, device=llm.device).unsqueeze(0) < attack_lengths.unsqueeze(1)

        @torch.no_grad()
        def _compute_y_logits(x: torch.Tensor) -> torch.Tensor:
            """
            Compute the logits that correspond to the prediction of the target suffix,
            given a prefix which may be the original prefix or the attack prefix.

            Args:
                x: The prefix tokens to be used for the forward pass.

            Returns:
                The logits for the target suffix, already flattened.
            """
            nonlocal llm, target_response_ids, prefix_len
            # Compute the forward pass with the given prefix
            llm_tokens = torch.cat([x, target_response_ids], dim=1)
            logits = llm(llm_tokens, return_dict=True).logits

            # Extract only the logits that should predict the target response
            return logits[:, prefix_len - 1:-1, :]

        def _compute_logits_ce_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            nonlocal target_response_ids, vocab_size, batch_size_actual, max_suffix_len
            # Compute the cross-entropy loss for every sample in the batch, considering only valid (non-padded) positions
            ce_loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_response_ids.view(-1),
                reduction='none',
            ).view(batch_size_actual, max_suffix_len)
            
            # Apply mask to ignore padded positions and compute mean only over valid positions
            masked_loss = ce_loss * mask.float()
            return masked_loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # Avoid division by zero

        with torch.no_grad():
            # Compute the forward pass for the original and attack prefixes
            original_logits = _compute_y_logits(original_prefix_ids)
            attack_logits = _compute_y_logits(x_attack_ids)

            # Compute the CE-loss for both, using appropriate masks
            overall_original_loss.append(_compute_logits_ce_loss(original_logits, target_mask))
            overall_attack_loss.append(_compute_logits_ce_loss(attack_logits, target_mask))

            # Compute the KL-divergence between the two, considering only valid positions
            kl_div = F.kl_div(
                attack_logits.reshape(-1, vocab_size).log_softmax(dim=-1),
                original_logits.reshape(-1, vocab_size).log_softmax(dim=-1),
                reduction='none',
                log_target=True,
            ).sum(dim=-1).view(batch_size_actual, max_suffix_len)
            
            # Apply mask and compute mean over valid positions
            masked_kl_div = kl_div * target_mask.float()
            kl_div_mean_per_sample = masked_kl_div.sum(dim=-1) / target_mask.sum(dim=-1).clamp(min=1)
            overall_kl_div.append(kl_div_mean_per_sample)

    # Compute the mean of the losses
    original_loss = torch.cat(overall_original_loss)
    original_loss_mean, original_loss_stddev = original_loss.mean(), original_loss.std()

    attack_loss = torch.cat(overall_attack_loss)
    attack_loss_mean, attack_loss_stddev = attack_loss.mean(), attack_loss.std()

    kl_div = torch.cat(overall_kl_div)
    kl_div_mean, kl_div_stddev = kl_div.mean(), kl_div.std()

    # Log attack loss metrics
    wandb.log({
        "attack_losses/original_loss_mean": original_loss_mean.item(),
        "attack_losses/original_loss_stddev": original_loss_stddev.item(),
        "attack_losses/attack_loss_mean": attack_loss_mean.item(),
        "attack_losses/attack_loss_stddev": attack_loss_stddev.item(),
        "attack_losses/kl_divergence_mean": kl_div_mean.item(),
        "attack_losses/kl_divergence_stddev": kl_div_stddev.item(),
        "attack_losses/loss_reduction": (original_loss_mean - attack_loss_mean).item(),
        "attack_losses/loss_reduction_ratio": ((original_loss_mean - attack_loss_mean) / original_loss_mean).item() if original_loss_mean > 0 else 0
    })

    return original_loss_mean, original_loss_stddev, attack_loss_mean, attack_loss_stddev, kl_div_mean, kl_div_stddev


def analyze_gcg_results(lightning_model: BaseLMModel, gcg_output_file: pathlib.Path, batch_size: int = 128):
    with gcg_output_file.open('r') as f:
        gcg_results = [gcg_evaluation.GCGResult.from_dict(r) for r in json.load(f)]

    token_attack_success_rate = compute_success_rate(gcg_results)
    print(f'Token attack success rate: {token_attack_success_rate:.2%}')

    successful_gcg_results = [result for result in gcg_results if result.get_success_tokens() > 1]

    mean_success_steps, stddev_success_steps = compute_mean_steps_to_success(successful_gcg_results)
    print(f'Mean steps to success: {mean_success_steps:.0f} ± {stddev_success_steps:.0f}')

    # Compute the attack losses
    original_loss_mean, original_loss_stddev, attack_loss_mean, attack_loss_stddev, kl_div_mean, \
        kl_div_stddev = compute_attack_losses(successful_gcg_results, lightning_model, batch_size=batch_size)
    print(f'Mean original X CE-loss: {original_loss_mean:.2f} ± {original_loss_stddev:.2f}')
    print(f'Mean attack X CE-loss: {attack_loss_mean:.2f} ± {attack_loss_stddev:.2f}')
    print(f'Mean KL-divergence between original and attack Xs: {kl_div_mean:.2f} ± {kl_div_stddev:.2f}')
    
    # Additional analysis and logging
    sample_level_success_rate = len(successful_gcg_results) / len(gcg_results) if gcg_results else 0
    
    # Log sample-level success metrics
    wandb.log({
        "success_metrics/sample_success_rate": sample_level_success_rate,
        "success_metrics/successful_samples": len(successful_gcg_results),
        "success_metrics/total_samples": len(gcg_results)
    })
    
    # Create and log summary table
    if wandb.run is not None:
        results_summary = wandb.Table(
            columns=["Metric", "Value", "Std Dev"],
            data=[
                ["Token Success Rate", f"{token_attack_success_rate:.4f}", "N/A"],
                ["Sample Success Rate", f"{sample_level_success_rate:.4f}", "N/A"],
                ["Mean Steps to Success", f"{mean_success_steps:.2f}", f"{stddev_success_steps:.2f}"],
                ["Original Loss", f"{original_loss_mean:.4f}", f"{original_loss_stddev:.4f}"],
                ["Attack Loss", f"{attack_loss_mean:.4f}", f"{attack_loss_stddev:.4f}"],
                ["KL Divergence", f"{kl_div_mean:.4f}", f"{kl_div_stddev:.4f}"],
            ]
        )
        wandb.log({"results_summary": results_summary})


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    """
    Main function to train the model with the Inverse First Token task.

    Args:
        cfg: Configuration object with all parameters
    """
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Initialize wandb
    run_name = f"gcg_attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        entity="pag-llm-team",
        project="pag-llm-gcg-attacks",
        name=run_name,
        config={
            **dataclasses.asdict(cfg),
        },
        tags=["gcg", "adversarial_attack", "language_model", "security"]
    )

    # Instantiate model and data module
    lightning_model, data_module, model_name, cfg = load_model_from_checkpoint(
        cfg.model.checkpoint_path,
        cfg,
    )

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.training.device is not None:
        torch_device = f'cuda:{cfg.training.device[0]}'
    lightning_model.to(torch_device).eval()
    
    # Log model information
    total_params = sum(p.numel() for p in lightning_model.parameters())
    trainable_params = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)
    wandb.log({
        "model_info/total_parameters": total_params,
        "model_info/trainable_parameters": trainable_params,
        "model_info/model_name": model_name,
        "model_info/device": str(torch_device)
    })

    # Run GCG
    gcg = gcg_algorithm.GCG(
        model=lightning_model.model,
        tokenizer=lightning_model.tokenizer,
        num_prefix_tokens=20, # GCG original work uses 20
        num_steps=500, # GCG original work use 500
        search_width=512, # GCG original work uses 512 as "batch size"
        top_k=256, # GCG original work uses 256
    )
    # run_gcg_single_attack(gcg, target_response=' and it was a sunny day.')

    gcg_output_file = cfg.model.output_dir / f'gcg_{model_name}.json'
    if gcg_output_file.exists():
        print(f"File {gcg_output_file} already exists. Skipping GCG evaluation.")
    else:
        run_full_gcg_evaluation(gcg, data_module.test_dataset, gcg_output_file)
        
        analyze_gcg_results(lightning_model, gcg_output_file)
        
        # Log the output file as an artifact
        artifact = wandb.Artifact(
            name=f"gcg_results_{model_name}",
            type="results",
            description=f"GCG attack results for model {model_name}"
        )
        artifact.add_file(str(gcg_output_file))
        wandb.log_artifact(artifact)
    
        print(f"Experiment logged to wandb: {wandb.run.url}")
    wandb.finish()


if __name__ == '__main__':
    main()
