import dataclasses
import json
import pathlib
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import wandb
from sentence_transformers import SentenceTransformer

from config import CustomLLMPagConfig, apply_config
from data.data_processor import TextDataset
from gcg import gcg_algorithm, gcg_evaluation
from instantiate import load_model_from_checkpoint
from models.base_model import BaseLMModel
from infer_backward_tinystories import load_semantic_model, compute_semantic_similarity, get_batch_perplexity


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
                                                         max_samples_to_attack=1000,
                                                         random_select_samples=True,
                                                         evaluate_every_n_steps=500)
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
    
    return gcg_results


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
        target_len = target_response_ids.shape[1]
        target_lengths = torch.tensor([len(result.target_response_ids) for result in batch], device=llm.device)
        target_mask = torch.arange(target_len, device=llm.device).unsqueeze(0) < target_lengths.unsqueeze(1)

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
            return logits[:, prefix_len - 1:prefix_len - 1 + target_len, :]

        def _compute_logits_ce_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            nonlocal target_response_ids, vocab_size, batch_size_actual, max_suffix_len, pad_token_id
            # Compute the cross-entropy loss for every sample in the batch, considering only valid (non-padded) positions
            ce_loss = F.cross_entropy(
                logits.transpose(1, 2),  # (batch, vocab, seq)
                target_response_ids,
                reduction='none',
                ignore_index=pad_token_id
            )  # (batch, seq)
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
            ).sum(dim=-1).view(batch_size_actual, target_len)

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


def run_gcg_with_convergence_logging(gcg: gcg_algorithm.GCG, dataset: TextDataset, 
                                     gcg_output_file: pathlib.Path, 
                                     evaluate_every_n_steps: int = 500,
                                     lightning_model: BaseLMModel = None,
                                     cfg: CustomLLMPagConfig = None):
    """
    Run GCG evaluation with intermediate logging at specified step intervals.
    This runs GCG once for the maximum steps and collects results at intermediate intervals.
    
    Args:
        gcg: GCG algorithm instance
        dataset: Dataset to evaluate on
        gcg_output_file: Path to save final results
        evaluate_every_n_steps: Number of steps between evaluations (default: 500)
        lightning_model: Language model for computing naturalness metrics
        cfg: Configuration object for loading semantic model
    """
    print('Attacking:', gcg_output_file.stem, 'on', gcg.device)
    
    # Initialize semantic model for naturalness metrics if cfg provided
    semantic_model = None
    if cfg is not None:
        print("Loading SentenceTransformer model for naturalness metrics...")
        semantic_model = load_semantic_model(cfg)
    
    # Log GCG configuration
    wandb.log({
        "gcg_config/num_prefix_tokens": gcg.num_prefix_tokens,
        "gcg_config/num_steps": gcg.num_steps,
        "gcg_config/search_width": gcg.search_width,
        "gcg_config/top_k": gcg.top_k,
        "gcg_config/device": str(gcg.device),
        "gcg_config/dataset_size": len(dataset),
        "gcg_config/evaluate_every_n_steps": evaluate_every_n_steps
    })
    
    start_time = time.time()
    
    # Run single evaluation with intermediate logging
    gcg_results = gcg_evaluation.evaluate_model_with_gcg(
        gcg, 
        dataset,
        target_response_len=10,
        max_samples_to_attack=None,
        random_select_samples=True,
        evaluate_every_n_steps=evaluate_every_n_steps,
        lightning_model=lightning_model,
        semantic_model=semantic_model
    )
    
    evaluation_time = time.time() - start_time
    
    # Log evaluation time and basic stats
    wandb.log({
        "evaluation/total_time": evaluation_time,
        "evaluation/samples_attacked": len(gcg_results),
        "evaluation/avg_time_per_sample": evaluation_time / len(gcg_results) if gcg_results else 0
    })
    
    # Save final results
    with gcg_output_file.open('w') as f:
        json.dump([r.to_dict() for r in gcg_results], f, indent=4)
    print(f"Saved GCG results to {gcg_output_file}")
    
    return gcg_results


def log_convergence_metrics(gcg_results: list[gcg_evaluation.GCGResult], step_interval: int, 
                           lightning_model: BaseLMModel = None, semantic_model: SentenceTransformer = None):
    """Log convergence metrics for a specific step interval."""
    # Compute success rate
    num_successful_tokens = sum(
        result.get_success_tokens()
        for result in gcg_results
    )
    num_total_tokens = sum(
        min(len(result.target_response_ids), len(result.y_attack_response_ids))
        for result in gcg_results
    )
    token_success_rate = num_successful_tokens / num_total_tokens if num_total_tokens > 0 else 0
    
    # Compute sample-level success rate (at least 2 successful tokens)
    successful_samples = [r for r in gcg_results if r.get_success_tokens() >= 2]
    sample_success_rate = len(successful_samples) / len(gcg_results) if gcg_results else 0
    
    # Log basic metrics with step interval prefix
    prefix = f"convergence_{step_interval}steps"
    basic_metrics = {
        f"{prefix}/token_success_rate": token_success_rate,
        f"{prefix}/sample_success_rate": sample_success_rate,
        f"{prefix}/successful_tokens": num_successful_tokens,
        f"{prefix}/total_tokens": num_total_tokens,
        f"{prefix}/successful_samples": len(successful_samples),
        f"{prefix}/total_samples": len(gcg_results),
        f"{prefix}/steps": step_interval
    }
    
    # Compute and log naturalness metrics if models are provided
    if lightning_model is not None and semantic_model is not None:
        naturalness_metrics = compute_naturalness_metrics(
            gcg_results, lightning_model, semantic_model, step_interval
        )
        basic_metrics.update(naturalness_metrics)
    
    wandb.log(basic_metrics)
    
    print(f"Step {step_interval}: Token success rate: {token_success_rate:.2%}, "
          f"Sample success rate: {sample_success_rate:.2%}")
    
    if lightning_model is not None and semantic_model is not None:
        print(f"  Naturalness metrics:")
        print(f"    Mean prefix perplexity: {basic_metrics.get(f'naturalness_{step_interval}steps/mean_prefix_perplexity', 0):.2f}")
        print(f"    Mean semantic similarity: {basic_metrics.get(f'naturalness_{step_interval}steps/mean_semantic_similarity', 0):.4f}")
        print(f"    Mean non-alphanumeric ratio: {basic_metrics.get(f'naturalness_{step_interval}steps/mean_non_alphanumeric_ratio', 0):.4f}")
        print(f"    Mean max consecutive repetition: {basic_metrics.get(f'naturalness_{step_interval}steps/mean_max_consecutive_repetition', 0):.2f}")


def create_convergence_chart(log_intervals: list[int]):
    """Create a convergence chart showing success rates over steps."""
    if wandb.run is not None:
        # Prepare data for convergence chart
        steps = []
        token_success_rates = []
        sample_success_rates = []
        
        for interval in log_intervals:
            steps.append(interval)
            # Get metrics from wandb logs
            try:
                token_rate = wandb.run.summary.get(f"convergence_{interval}steps/token_success_rate", 0)
                sample_rate = wandb.run.summary.get(f"convergence_{interval}steps/sample_success_rate", 0)
                token_success_rates.append(token_rate)
                sample_success_rates.append(sample_rate)
            except:
                token_success_rates.append(0)
                sample_success_rates.append(0)
        
        # Create convergence table
        convergence_data = []
        for i, step in enumerate(steps):
            convergence_data.append([
                step,
                f"{token_success_rates[i]:.4f}",
                f"{sample_success_rates[i]:.4f}"
            ])
        
        convergence_table = wandb.Table(
            columns=["Steps", "Token Success Rate", "Sample Success Rate"],
            data=convergence_data
        )
        
        wandb.log({
            "convergence_analysis": convergence_table,
            "convergence_chart": wandb.plot.line(
                convergence_table, 
                "Steps", 
                "Token Success Rate",
                title="GCG Convergence Analysis"
            )
        })

@torch.no_grad()
def count_repeated_tokens(x_input_ids: torch.Tensor, x_attention_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Count the number of repeated tokens in the given input IDs, considering the attention mask for valid tokens.
    Adapted from infer_backward_tinystories.py
    
    Args:
        x_input_ids: Input IDs of the sentences.
        x_attention_mask: Attention mask of the sentences. If None, assumes all tokens are valid.

    Returns:
        torch.Tensor: The number of repeated tokens for each sentence.
    """
    if x_input_ids.ndim == 1:
        x_input_ids = x_input_ids.unsqueeze(0)
    
    if x_attention_mask is None:
        x_attention_mask = torch.ones_like(x_input_ids)
    elif x_attention_mask.ndim == 1:
        x_attention_mask = x_attention_mask.unsqueeze(0)
    
    assert x_input_ids.shape == x_attention_mask.shape, \
        f'input_ids and attention_mask shape mismatch: {x_input_ids.shape} != {x_attention_mask.shape}'
    assert x_input_ids.ndim == 2, \
        f'input_ids shape mismatch: {x_input_ids.shape} != (batch_size, seq_len)'

    num_classes = x_input_ids.max().item() + 1

    # Set an invalid token ID where the attention_mask is zero
    invalid_token_id = num_classes
    x_input_ids = x_input_ids.masked_fill(x_attention_mask == 0, invalid_token_id)

    # Get the counts of each token in the batch, keeping the batch dimension
    target = torch.zeros(x_input_ids.size(0), invalid_token_id + 1, dtype=x_input_ids.dtype, device=x_input_ids.device)
    values = torch.ones_like(x_input_ids)
    target.scatter_add_(dim=1, index=x_input_ids, src=values)

    # Remove the invalid token id from the target
    target = target[:, :invalid_token_id]

    # Now, remove zeros (tokens not showing in the input_ids) and ones (tokens not repeating) from the target
    target = target.masked_fill_(target < 2, 1)
    total_repeated_tokens = (target - 1).sum(dim=1)
    return total_repeated_tokens


def compute_non_alphanumeric_ratio(text: str) -> float:
    """
    Compute the ratio of non-alphanumeric characters in the text.
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of non-alphanumeric characters (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    non_alphanumeric_count = sum(1 for char in text if not char.isalnum())
    return non_alphanumeric_count / len(text)


def compute_max_consecutive_token_repetition(input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> int:
    """
    Compute the maximum consecutive token repetition in the input sequence.
    
    Args:
        input_ids: Input token IDs (1D tensor)
        attention_mask: Attention mask (1D tensor). If None, assumes all tokens are valid.
        
    Returns:
        int: Maximum consecutive token repetition length
    """
    if input_ids.ndim > 1:
        input_ids = input_ids.squeeze()
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    elif attention_mask.ndim > 1:
        attention_mask = attention_mask.squeeze()
    
    # Get valid tokens only
    valid_tokens = input_ids[attention_mask == 1]
    
    if len(valid_tokens) <= 1:
        return 1
    
    max_consecutive = 1
    current_consecutive = 1
    
    for i in range(1, len(valid_tokens)):
        if valid_tokens[i] == valid_tokens[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    
    return max_consecutive


def compute_naturalness_metrics(gcg_results: list[gcg_evaluation.GCGResult], 
                               lightning_model: BaseLMModel,
                               semantic_model: SentenceTransformer,
                               step_interval: int,
                               batch_size: int = 128) -> dict:
    """
    Compute naturalness metrics for GCG-generated prefixes.
    
    Args:
        gcg_results: List of GCG results
        lightning_model: The language model for computing perplexity
        semantic_model: SentenceTransformer model for semantic similarity
        step_interval: The step interval for logging
        batch_size: Batch size for processing
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    if not gcg_results:
        return {}
    
    tokenizer = lightning_model.tokenizer
    device = lightning_model.device
    
    # Collect all metrics
    prefix_perplexities = []
    total_token_repetitions = []
    semantic_similarities = []
    non_alphanumeric_ratios = []
    max_consecutive_repetitions = []
    
    # Process in batches
    for start_i in range(0, len(gcg_results), batch_size):
        end_i = min(start_i + batch_size, len(gcg_results))
        batch = gcg_results[start_i:end_i]
        
        # Handle tokenizers that might not have a pad token
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Prepare batch data
        original_prefix_ids = pad_sequence([torch.tensor(result.original_prefix_ids) for result in batch], 
                                         batch_first=True, padding_value=pad_token_id).to(device)
        attack_prefix_ids = []
        
        # Extract attack prefixes from attack strings
        for result in batch:
            attack_tokens = tokenizer.encode(result.x_attack_str, return_tensors='pt').squeeze()
            if attack_tokens.ndim == 0:
                attack_tokens = attack_tokens.unsqueeze(0)
            attack_prefix_ids.append(attack_tokens)
        
        attack_prefix_ids = pad_sequence(attack_prefix_ids, batch_first=True, padding_value=pad_token_id).to(device)
        
        # Create attention masks
        original_attention_mask = (original_prefix_ids != pad_token_id).long()
        attack_attention_mask = (attack_prefix_ids != pad_token_id).long()
        
        # Compute perplexity for attack prefixes
        attack_perplexities = get_batch_perplexity(lightning_model, attack_prefix_ids, attack_attention_mask)
        prefix_perplexities.extend(attack_perplexities.cpu().tolist())
        
        # Compute token repetition for attack prefixes
        attack_repetitions = count_repeated_tokens(attack_prefix_ids, attack_attention_mask)
        total_token_repetitions.extend(attack_repetitions.cpu().tolist())
        
        # Compute semantic similarity and character-level metrics
        for i, result in enumerate(batch):
            # Get original and attack prefix texts
            original_text = tokenizer.decode(result.original_prefix_ids, skip_special_tokens=True)
            attack_text = result.x_attack_str
            
            # Semantic similarity
            semantic_sim = compute_semantic_similarity(attack_text, original_text, semantic_model)
            semantic_similarities.append(semantic_sim)
            
            # Non-alphanumeric ratio
            non_alpha_ratio = compute_non_alphanumeric_ratio(attack_text)
            non_alphanumeric_ratios.append(non_alpha_ratio)
            
            # Maximum consecutive token repetition
            max_consecutive = compute_max_consecutive_token_repetition(
                attack_prefix_ids[i], attack_attention_mask[i]
            )
            max_consecutive_repetitions.append(max_consecutive)
    
    # Compute aggregate metrics
    metrics = {
        f"naturalness_{step_interval}steps/mean_prefix_perplexity": sum(prefix_perplexities) / len(prefix_perplexities),
        f"naturalness_{step_interval}steps/std_prefix_perplexity": torch.tensor(prefix_perplexities, dtype=torch.float32).std().item(),

        f"naturalness_{step_interval}steps/mean_total_token_repetition": sum(total_token_repetitions) / len(total_token_repetitions),
        f"naturalness_{step_interval}steps/std_total_token_repetition": torch.tensor(total_token_repetitions, dtype=torch.float32).std().item(),

        f"naturalness_{step_interval}steps/mean_semantic_similarity": sum(semantic_similarities) / len(semantic_similarities),
        f"naturalness_{step_interval}steps/std_semantic_similarity": torch.tensor(semantic_similarities, dtype=torch.float32).std().item(),

        f"naturalness_{step_interval}steps/mean_non_alphanumeric_ratio": sum(non_alphanumeric_ratios) / len(non_alphanumeric_ratios),
        f"naturalness_{step_interval}steps/std_non_alphanumeric_ratio": torch.tensor(non_alphanumeric_ratios, dtype=torch.float32).std().item(),

        f"naturalness_{step_interval}steps/mean_max_consecutive_repetition": sum(max_consecutive_repetitions) / len(max_consecutive_repetitions),
        f"naturalness_{step_interval}steps/std_max_consecutive_repetition": torch.tensor(max_consecutive_repetitions, dtype=torch.float32).std().item(),

        f"naturalness_{step_interval}steps/num_samples": len(gcg_results),
    }
    
    return metrics


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
    num_steps = 500
    gcg = gcg_algorithm.GCG(
        model=lightning_model.model,
        tokenizer=lightning_model.tokenizer,
        num_prefix_tokens=20, # GCG original work uses 20
        num_steps=num_steps, # Extended to num_steps for convergence analysis
        search_width=16384, # GCG original work uses 512 as "batch size"
        top_k=256, # GCG original work uses 256
    )
    # run_gcg_single_attack(gcg, target_response=' and it was a sunny day.')

    gcg_output_file = cfg.model.output_dir / f'gcg_{model_name}.json'
    if gcg_output_file.exists():
        print(f"File {gcg_output_file} already exists. Skipping GCG evaluation.")
    else:
        # Use regular intervals for convergence logging
        evaluate_every_n_steps = 125
        print(f"Running GCG evaluation with convergence logging every {evaluate_every_n_steps} steps")
        
        gcg_results = run_gcg_with_convergence_logging(
            gcg, 
            data_module.test_dataset, 
            gcg_output_file, 
            evaluate_every_n_steps=evaluate_every_n_steps,
            lightning_model=lightning_model,
            cfg=cfg
        )
        
    # Analyze final results
    analyze_gcg_results(lightning_model, gcg_output_file)
    
    # Log the output file as an artifact
    artifact = wandb.Artifact(
        name=f"gcg_results_{model_name}",
        type="results",
        description=f"GCG attack results for model {model_name} with convergence analysis"
    )
    artifact.add_file(str(gcg_output_file))
    wandb.log_artifact(artifact)

    print(f"Experiment logged to wandb: {wandb.run.url}")
    wandb.finish()


if __name__ == '__main__':
    main()
