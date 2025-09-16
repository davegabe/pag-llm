import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from lightning.pytorch.loggers import WandbLogger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

from config import CustomLLMPagConfig, apply_config
from evaluation_metrics import BackwardInferenceEvaluator, aggregate_metrics
from infer_backward_tinystories import (
    load_semantic_model, 
    compute_semantic_similarity,
    get_batch_perplexity
)
from inverse_lm_stats import init_evaluation
from models.base_model import BaseLMModel


def load_backward_inference_results(results_files: list[str]) -> list[dict]:
    """
    Load backward inference results from multiple worker files.
    
    Args:
        results_files: List of JSON file paths containing worker results
        
    Returns:
        list[dict]: Combined results from all workers
    """
    all_results = []
    
    for file_path in results_files:
        if not Path(file_path).exists():
            print(f"Warning: Results file not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r') as f:
                worker_results = json.load(f)
                all_results.extend(worker_results)
                print(f"Loaded {len(worker_results)} results from {file_path}")
        except Exception as e:
            print(f"Error loading results from {file_path}: {e}")
            continue
    
    print(f"Total loaded results: {len(all_results)}")
    return all_results


def prepare_batched_data(results: List[Dict], batch_size: int, device: torch.device) -> List[Dict]:
    """
    Prepare batched data for efficient processing.
    
    Args:
        results: List of individual results
        batch_size: Size of batches to create
        device: Device to move tensors to
        
    Returns:
        List of batched data dictionaries
    """
    print(f"Preparing batched data with batch_size={batch_size}...")
    
    batched_data = []
    
    for i in range(0, len(results), batch_size):
        batch_results = results[i:i + batch_size]
        
        # Collect all texts for this batch
        original_texts = [r['original_text'] for r in batch_results]
        predicted_texts = [r['predicted_text'] for r in batch_results]
        
        # Prepare tensor data
        batch_data = {
            'batch_results': batch_results,
            'original_texts': original_texts,
            'predicted_texts': predicted_texts,
            'batch_size': len(batch_results),
            'start_idx': i,
            'end_idx': i + len(batch_results)
        }
        
        # Convert token data to tensors and pad if needed
        max_orig_len = max(len(r['original_input_ids']) for r in batch_results)
        max_pred_len = max(len(r['predicted_input_ids']) for r in batch_results)
        
        original_input_ids_batch = []
        original_attention_mask_batch = []
        predicted_input_ids_batch = []
        predicted_attention_mask_batch = []
        
        for result in batch_results:
            # Original data
            orig_ids = result['original_input_ids'] + [0] * (max_orig_len - len(result['original_input_ids']))
            orig_mask = result['original_attention_mask'] + [0] * (max_orig_len - len(result['original_attention_mask']))
            original_input_ids_batch.append(orig_ids)
            original_attention_mask_batch.append(orig_mask)
            
            # Predicted data
            pred_ids = result['predicted_input_ids'] + [0] * (max_pred_len - len(result['predicted_input_ids']))
            pred_mask = result['predicted_attention_mask'] + [0] * (max_pred_len - len(result['predicted_attention_mask']))
            predicted_input_ids_batch.append(pred_ids)
            predicted_attention_mask_batch.append(pred_mask)
        
        # Convert to tensors and move to device
        batch_data.update({
            'original_input_ids': torch.tensor(original_input_ids_batch, dtype=torch.long, device=device),
            'original_attention_mask': torch.tensor(original_attention_mask_batch, dtype=torch.long, device=device),
            'predicted_input_ids': torch.tensor(predicted_input_ids_batch, dtype=torch.long, device=device),
            'predicted_attention_mask': torch.tensor(predicted_attention_mask_batch, dtype=torch.long, device=device),
        })
        
        batched_data.append(batch_data)
    
    print(f"Created {len(batched_data)} batches")
    return batched_data


def compute_batched_perplexities(batched_data: List[Dict], lightning_model: BaseLMModel) -> tuple[List[float], List[float]]:
    """
    Compute perplexities for all samples in batch mode.
    
    Args:
        batched_data: List of batched data
        lightning_model: The model to use for perplexity computation
        
    Returns:
        Tuple of (predicted_ppls, original_ppls) for each sample
    """
    print("Computing batched perplexities...")
    
    all_predicted_ppls = []
    all_original_ppls = []
    
    lightning_model.eval()
    with torch.no_grad():
        for batch_data in tqdm(batched_data, desc="Computing perplexities"):
            # Compute predicted perplexities
            predicted_ppls = get_batch_perplexity(
                lightning_model,
                batch_data['predicted_input_ids'],
                batch_data['predicted_attention_mask']
            )
            all_predicted_ppls.extend(predicted_ppls.cpu().tolist())
            
            # Compute original perplexities
            original_ppls = get_batch_perplexity(
                lightning_model,
                batch_data['original_input_ids'],
                batch_data['original_attention_mask']
            )
            all_original_ppls.extend(original_ppls.cpu().tolist())
    
    return all_predicted_ppls, all_original_ppls


def compute_batched_semantic_similarities(batched_data: List[Dict], semantic_model: SentenceTransformer) -> List[float]:
    """
    Compute semantic similarities for all samples in batch mode.
    
    Args:
        batched_data: List of batched data
        semantic_model: Sentence transformer model
        
    Returns:
        List of semantic similarity values for each sample
    """
    print("Computing batched semantic similarities...")
    
    all_similarities = []
    
    for batch_data in tqdm(batched_data, desc="Computing semantic similarities"):
        # Compute embeddings for the entire batch
        original_texts = batch_data['original_texts']
        predicted_texts = batch_data['predicted_texts']
        
        # Encode in batches for efficiency
        original_embeddings = semantic_model.encode(original_texts, convert_to_tensor=True)
        predicted_embeddings = semantic_model.encode(predicted_texts, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            original_embeddings, predicted_embeddings, dim=1
        )
        
        all_similarities.extend(similarities.cpu().tolist())
    
    return all_similarities


def compute_batched_third_party_perplexities(batched_data: List[Dict], external_llm: str) -> tuple[List[float], List[float]]:
    """
    Compute third-party perplexities for all samples in batch mode.
    
    Args:
        batched_data: List of batched data
        external_llm: External LLM model name/path
        
    Returns:
        Tuple of (predicted_ppls, original_ppls)
    """
    print("Computing third-party perplexities...")
    
    import evaluate
    perplexity_evaluator = evaluate.load("perplexity", module_type="metric")
    
    all_predicted_texts = []
    all_original_texts = []
    
    # Collect all texts and limit to 50 words each
    for batch_data in batched_data:
        for pred_text, orig_text in zip(batch_data['predicted_texts'], batch_data['original_texts']):
            pred_limited = ' '.join(pred_text.split()[:50])
            orig_limited = ' '.join(orig_text.split()[:50])
            all_predicted_texts.append(pred_limited)
            all_original_texts.append(orig_limited)
    
    # Compute perplexities for all texts at once
    all_texts = all_predicted_texts + all_original_texts
    
    print(f"Computing perplexities for {len(all_texts)} texts using {external_llm}...")
    
    results = perplexity_evaluator.compute(
        model_id=external_llm,
        predictions=all_texts,
        add_start_token=False,
        batch_size=16  # Process in smaller batches to avoid memory issues
    )
    
    perplexities = results['perplexities'] if results is not None else [float('inf')] * len(all_texts)
    
    # Split back into predicted and original
    mid_point = len(all_predicted_texts)
    predicted_ppls = perplexities[:mid_point]
    original_ppls = perplexities[mid_point:]
    
    return predicted_ppls, original_ppls


def compute_comprehensive_metrics_individual(
    results: List[Dict],
    predicted_ppls: List[float],
    original_ppls: List[float],
    semantic_similarities: List[float],
    third_party_predicted_ppls: List[float],
    third_party_original_ppls: List[float],
    lightning_model: BaseLMModel,
    evaluator: BackwardInferenceEvaluator
) -> List[Dict]:
    """
    Compute remaining individual metrics that require per-sample processing.
    
    Args:
        results: List of individual results
        predicted_ppls: Pre-computed predicted perplexities
        original_ppls: Pre-computed original perplexities
        semantic_similarities: Pre-computed semantic similarities
        third_party_predicted_ppls: Pre-computed third-party predicted perplexities
        third_party_original_ppls: Pre-computed third-party original perplexities
        lightning_model: Main model
        evaluator: Evaluator instance
        
    Returns:
        List of comprehensive metrics for each sample
    """
    print("Computing remaining individual metrics...")
    
    comprehensive_sample_metrics = []
    device = lightning_model.device
    
    for i, result in enumerate(tqdm(results, desc="Individual metrics")):
        try:
            # Convert to tensors
            original_input_ids = torch.tensor(result['original_input_ids'], dtype=torch.long, device=device)
            original_attention_mask = torch.tensor(result['original_attention_mask'], dtype=torch.long, device=device)
            predicted_input_ids = torch.tensor(result['predicted_input_ids'], dtype=torch.long, device=device)
            predicted_attention_mask = torch.tensor(result['predicted_attention_mask'], dtype=torch.long, device=device)
            
            # Extract prefix and suffix data
            prefix_len = result['prefix_len']
            original_prefix_ids = original_input_ids[:prefix_len]
            predicted_prefix_ids = predicted_input_ids[:prefix_len]
            suffix_ids = torch.tensor(result['suffix_input_ids'], dtype=torch.long, device=device)
            
            original_prefix_mask = original_attention_mask[:prefix_len]
            predicted_prefix_mask = predicted_attention_mask[:prefix_len]
            suffix_mask = torch.tensor(result['suffix_attention_mask'], dtype=torch.long, device=device)
            
            # Compute token overlap metrics
            overlap_metrics = evaluator.compute_token_overlap_metrics(
                original_prefix_ids, predicted_prefix_ids,
                original_prefix_mask, predicted_prefix_mask
            )
            
            # Compute positional accuracy metrics
            positional_metrics = evaluator.compute_positional_accuracy(
                original_prefix_ids, predicted_prefix_ids,
                original_prefix_mask, predicted_prefix_mask
            )
            
            # Compute forward coherence metrics
            coherence_metrics = evaluator.compute_forward_coherence(
                predicted_prefix_ids, suffix_ids,
                predicted_prefix_mask, suffix_mask
            )
            
            # Combine all metrics
            sample_metrics = {
                **overlap_metrics,
                **positional_metrics,
                **coherence_metrics,
                'predicted_perplexity': third_party_predicted_ppls[i],
                'original_perplexity': third_party_original_ppls[i],
                # Add the pre-computed batch metrics as additional references
                'batch_predicted_ppl': predicted_ppls[i],
                'batch_original_ppl': original_ppls[i],
                'semantic_similarity': semantic_similarities[i],
            }
            
            comprehensive_sample_metrics.append(sample_metrics)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # Add empty metrics to maintain alignment
            comprehensive_sample_metrics.append({})
            continue
    
    return comprehensive_sample_metrics


def compute_backward_inference_metrics(
        results: list[dict],
        lightning_model: BaseLMModel,
        baseline_model: BaseLMModel | None,
        semantic_model: SentenceTransformer,
        evaluator: BackwardInferenceEvaluator,
        batch_size: int = 32,
        output_dir: Path | None = None,
        model_name: str | None = None,
        use_init: str | None = None
) -> tuple[list[dict], dict]:
    """
    Compute comprehensive metrics for backward inference results using optimized batching.
    
    Args:
        results: List of backward inference results
        lightning_model: Main trained model
        baseline_model: Baseline model for comparison (optional)
        semantic_model: Sentence transformer for semantic similarity
        evaluator: Backward inference evaluator
        batch_size: Batch size for processing
        output_dir: Directory to save best samples (optional)
        model_name: Model name for file naming (optional)
        use_init: Initialization strategy for file naming (optional)
        
    Returns:
        tuple: (sample_metrics, aggregate_metrics)
    """
    print("Computing comprehensive metrics for backward inference results (OPTIMIZED)...")
    
    device = lightning_model.device
    
    # Prepare batched data
    batched_data = prepare_batched_data(results, batch_size, device)
    
    # Compute all perplexities in batch mode
    predicted_ppls, original_ppls = compute_batched_perplexities(batched_data, lightning_model)
    
    # Compute all semantic similarities in batch mode
    semantic_similarities = compute_batched_semantic_similarities(batched_data, semantic_model)
    
    # Compute third-party perplexities in batch mode
    external_llm_path = str(evaluator.external_llm)
    third_party_predicted_ppls, third_party_original_ppls = compute_batched_third_party_perplexities(
        batched_data, external_llm_path
    )
    
    # Compute remaining individual metrics
    comprehensive_sample_metrics = compute_comprehensive_metrics_individual(
        results, predicted_ppls, original_ppls, semantic_similarities,
        third_party_predicted_ppls, third_party_original_ppls,
        lightning_model, evaluator
    )
    
    # Compute aggregate metrics
    actual_samples = len(results)
    avg_predicted_ppl = np.mean(predicted_ppls)
    avg_original_ppl = np.mean(original_ppls)
    avg_semantic_similarity = np.mean(semantic_similarities)
    avg_third_party_predicted_ppl = np.mean(third_party_predicted_ppls)
    avg_third_party_original_ppl = np.mean(third_party_original_ppls)
    
    # Aggregate comprehensive metrics
    aggregated_comprehensive_metrics = aggregate_metrics(comprehensive_sample_metrics)
    
    aggregate_final_metrics = {
        'avg_predicted_overall_ppl': avg_predicted_ppl,
        'avg_original_overall_ppl': avg_original_ppl,
        'avg_semantic_similarity': avg_semantic_similarity,
        'avg_third_party_predicted_ppl': avg_third_party_predicted_ppl,
        'avg_third_party_original_ppl': avg_third_party_original_ppl,
        'ppl_improvement': avg_original_ppl - avg_predicted_ppl,
        'ppl_improvement_ratio': avg_predicted_ppl / avg_original_ppl if avg_original_ppl > 0 else float('inf'),
        'third_party_ppl_improvement': avg_third_party_original_ppl - avg_third_party_predicted_ppl,
        'actual_samples_used': actual_samples,
    }
    
    # Add aggregated comprehensive metrics
    aggregate_final_metrics.update(aggregated_comprehensive_metrics)
    
    # Extract and save best samples based on third-party perplexity
    if output_dir and model_name and use_init:
        best_samples = extract_and_save_best_samples(
            results=results,
            third_party_predicted_ppls=third_party_predicted_ppls,
            third_party_original_ppls=third_party_original_ppls,
            semantic_similarities=semantic_similarities,
            output_dir=output_dir,
            model_name=model_name,
            use_init=use_init,
            top_k=10
        )
        
        # Add best samples info to aggregate metrics for WandB logging
        aggregate_final_metrics['num_best_samples_saved'] = len(best_samples)
        if best_samples:
            aggregate_final_metrics['best_sample_ppl'] = best_samples[0]['metrics']['third_party_predicted_ppl']
            aggregate_final_metrics['best_sample_improvement'] = best_samples[0]['metrics']['ppl_improvement']
            aggregate_final_metrics['best_sample_semantic_sim'] = best_samples[0]['metrics']['semantic_similarity']
    else:
        print("Warning: Missing parameters for saving best samples (output_dir, model_name, or use_init)")
        aggregate_final_metrics['num_best_samples_saved'] = 0
    
    if baseline_model is not None:
        # TODO: Add baseline comparison if needed
        pass
    
    print(f"Optimization summary:")
    print(f"  - Processed {actual_samples} samples")
    print(f"  - Used batch size: {batch_size}")
    print(f"  - Created {len(batched_data)} batches")
    print(f"  - Average PPL improvement: {avg_original_ppl - avg_predicted_ppl:.2f}")
    print(f"  - Average semantic similarity: {avg_semantic_similarity:.4f}")
    
    return comprehensive_sample_metrics, aggregate_final_metrics


def extract_and_save_best_samples(
    results: List[Dict],
    third_party_predicted_ppls: List[float],
    third_party_original_ppls: List[float],
    semantic_similarities: List[float],
    output_dir: Path,
    model_name: str,
    use_init: str,
    top_k: int = 10
) -> List[Dict]:
    """
    Extract and save the best performing samples based on third-party perplexity.
    
    Args:
        results: List of individual results
        third_party_predicted_ppls: Third-party predicted perplexities
        third_party_original_ppls: Third-party original perplexities
        semantic_similarities: Semantic similarities
        output_dir: Directory to save results
        model_name: Model name for file naming
        use_init: Initialization strategy for file naming
        top_k: Number of top samples to save
        
    Returns:
        List of best samples with additional metrics
    """
    print(f"\nExtracting top {top_k} best inverted prompts based on third-party perplexity...")
    
    # Combine all information for ranking
    sample_data = []
    for i, (result, pred_ppl, orig_ppl, sem_sim) in enumerate(zip(
        results, third_party_predicted_ppls, third_party_original_ppls, semantic_similarities
    )):
        ppl_improvement = orig_ppl - pred_ppl
        ppl_improvement_ratio = pred_ppl / orig_ppl if orig_ppl > 0 else float('inf')
        
        sample_data.append({
            'sample_idx': result.get('sample_idx', i),
            'original_text': result['original_text'],
            'predicted_text': result['predicted_text'],
            'prefix_len': result['prefix_len'],
            'suffix_length': result['suffix_length'],
            'use_init': result['use_init'],
            'beam_size': result['beam_size'],
            'third_party_predicted_ppl': pred_ppl,
            'third_party_original_ppl': orig_ppl,
            'ppl_improvement': ppl_improvement,
            'ppl_improvement_ratio': ppl_improvement_ratio,
            'semantic_similarity': sem_sim,
            # For ranking, we want lower predicted perplexity (better quality)
            'ranking_score': pred_ppl,
        })
    
    # Sort by predicted perplexity (lower is better)
    sample_data.sort(key=lambda x: x['ranking_score'])
    
    # Take top k samples
    best_samples = sample_data[:top_k]
    
    # Format for better readability
    formatted_best_samples = []
    for i, sample in enumerate(best_samples, 1):
        formatted_sample = {
            'rank': i,
            'sample_idx': sample['sample_idx'],
            'metrics': {
                'third_party_predicted_ppl': round(sample['third_party_predicted_ppl'], 4),
                'third_party_original_ppl': round(sample['third_party_original_ppl'], 4),
                'ppl_improvement': round(sample['ppl_improvement'], 4),
                'ppl_improvement_ratio': round(sample['ppl_improvement_ratio'], 4),
                'semantic_similarity': round(sample['semantic_similarity'], 4),
            },
            'generation_params': {
                'prefix_len': sample['prefix_len'],
                'suffix_length': sample['suffix_length'],
                'use_init': sample['use_init'],
                'beam_size': sample['beam_size'],
            },
            'texts': {
                'original_text': sample['original_text'],
                'predicted_text': sample['predicted_text'],
            }
        }
        formatted_best_samples.append(formatted_sample)
    
    # Add metadata about the evaluation run
    evaluation_metadata = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'use_init': use_init,
        'total_samples_evaluated': len(results),
        'top_k_selected': top_k,
        'ranking_criterion': 'third_party_predicted_ppl (lower is better)',
        'external_llm_used': 'Determined by evaluator configuration'
    }
    
    # Combine metadata and best samples
    output_data = {
        'metadata': evaluation_metadata,
        'best_samples': formatted_best_samples
    }
    
    # Save to JSON file
    best_samples_file = output_dir / f'best_inverted_prompts_{model_name}_{use_init}_top{top_k}.json'
    
    # Ensure output directory exists
    best_samples_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving best samples to: {best_samples_file}")
    
    with open(best_samples_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary of best samples
    print(f"\n=== TOP {top_k} BEST INVERTED PROMPTS ===")
    for i, sample in enumerate(formatted_best_samples[:5], 1):  # Show top 5 in console
        print(f"\nRank {i}:")
        print(f"  Sample Index: {sample['sample_idx']}")
        print(f"  Third-party PPL: {sample['metrics']['third_party_predicted_ppl']:.4f}")
        print(f"  PPL Improvement: {sample['metrics']['ppl_improvement']:.4f}")
        print(f"  Semantic Similarity: {sample['metrics']['semantic_similarity']:.4f}")
        print(f"  Original: {sample['texts']['original_text'][:100]}...")
        print(f"  Predicted: {sample['texts']['predicted_text'][:100]}...")
    
    if len(formatted_best_samples) > 5:
        print(f"\n... and {len(formatted_best_samples) - 5} more samples saved to file.")
    
    print(f"=====================================\n")
    
    return formatted_best_samples


def log_best_samples_to_wandb(wandb_logger: WandbLogger, output_dir: Path, model_name: str, use_init: str, top_k: int = 10):
    """
    Log the best samples to WandB as a table.
    
    Args:
        wandb_logger: WandB logger instance
        output_dir: Directory where best samples JSON file is saved
        model_name: Model name for file naming
        use_init: Initialization strategy for file naming
        top_k: Number of top samples to log
    """
    try:
        import wandb
        
        # Load best samples from the JSON file
        best_samples_file = output_dir / f'best_inverted_prompts_{model_name}_{use_init}_top{top_k}.json'
        
        if not best_samples_file.exists():
            print(f"Warning: Best samples file not found: {best_samples_file}")
            return
        
        with open(best_samples_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            best_samples = data.get('best_samples', data)  # Handle both old and new formats
        
        # Create WandB table
        columns = [
            "rank", "sample_idx", "third_party_ppl", "ppl_improvement", 
            "semantic_similarity", "prefix_len", "original_text", "predicted_text"
        ]
        
        table_data = []
        for sample in best_samples:
            # Truncate texts for better table display
            original_text = sample['texts']['original_text'][:200] + "..." if len(sample['texts']['original_text']) > 200 else sample['texts']['original_text']
            predicted_text = sample['texts']['predicted_text'][:200] + "..." if len(sample['texts']['predicted_text']) > 200 else sample['texts']['predicted_text']
            
            table_data.append([
                sample['rank'],
                sample['sample_idx'],
                sample['metrics']['third_party_predicted_ppl'],
                sample['metrics']['ppl_improvement'],
                sample['metrics']['semantic_similarity'],
                sample['generation_params']['prefix_len'],
                original_text,
                predicted_text
            ])
        
        # Log table to WandB
        table = wandb.Table(columns=columns, data=table_data)
        wandb_logger.experiment.log({f"best_inverted_prompts_top{top_k}": table})
        
        print(f"Successfully logged top {len(best_samples)} best samples to WandB")
        
    except Exception as e:
        print(f"Error logging best samples to WandB: {e}")


def print_evaluation_summary(metrics: dict, comprehensive_metrics: dict):
    """Print a comprehensive summary of evaluation results."""
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Samples Used: {metrics['actual_samples_used']}")
    print(f"Average Predicted PPL: {metrics['avg_predicted_overall_ppl']:.2f}")
    print(f"Average Original PPL: {metrics['avg_original_overall_ppl']:.2f}")
    print(f"Average Semantic Similarity: {metrics['avg_semantic_similarity']:.4f}")
    print(f"PPL Improvement: {metrics['ppl_improvement']:.2f}")
    print(f"PPL Improvement Ratio: {metrics['ppl_improvement_ratio']:.3f}")
    
    if 'avg_third_party_predicted_ppl' in metrics:
        print(f"\n--- Third-Party Model Metrics ---")
        print(f"Average Third-Party Predicted PPL: {metrics['avg_third_party_predicted_ppl']:.2f}")
        print(f"Average Third-Party Original PPL: {metrics['avg_third_party_original_ppl']:.2f}")
        print(f"Third-Party PPL Improvement: {metrics['third_party_ppl_improvement']:.2f}")
    
    # Print comprehensive metrics summary
    print(f"\n--- Token Overlap Metrics ---")
    print(f"Mean Token Precision: {comprehensive_metrics.get('mean_token_precision', 0.0):.4f}")
    print(f"Mean Token Recall: {comprehensive_metrics.get('mean_token_recall', 0.0):.4f}")
    print(f"Mean Token F1: {comprehensive_metrics.get('mean_token_f1', 0.0):.4f}")
    print(f"Mean Token Jaccard: {comprehensive_metrics.get('mean_token_jaccard', 0.0):.4f}")
    
    print(f"\n--- Position Accuracy Metrics ---")
    print(f"Mean Exact Match Accuracy: {comprehensive_metrics.get('mean_exact_match_accuracy', 0.0):.4f}")
    print(f"Mean Positional Accuracy: {comprehensive_metrics.get('mean_positional_accuracy', 0.0):.4f}")
    
    print(f"\n--- Forward Coherence Metrics ---")
    print(f"Mean Forward Coherence PPL: {comprehensive_metrics.get('mean_forward_coherence_ppl', float('inf')):.2f}")
    print(f"Mean Forward Coherence Loss: {comprehensive_metrics.get('mean_forward_coherence_loss', float('inf')):.4f}")

    if 'avg_bigram_overall_ppl' in metrics:
        print(f"\n--- Bigram Baseline Comparison ---")
        print(f"Average Bigram PPL: {metrics['avg_bigram_overall_ppl']:.2f}")
        print(f"Average Bigram Semantic Similarity: {metrics['avg_bigram_semantic_similarity']:.4f}")
        print(f"Bigram vs Predicted PPL Ratio: {metrics['bigram_vs_predicted_ppl_ratio']:.3f}")
        print(f"Semantic Similarity Improvement: {metrics['semantic_similarity_improvement']:.4f}")
    
    # Print best samples info if available
    if 'num_best_samples_saved' in metrics and metrics['num_best_samples_saved'] > 0:
        print(f"\n--- Best Samples Info ---")
        print(f"Number of Best Samples Saved: {metrics['num_best_samples_saved']}")
        if 'best_sample_ppl' in metrics:
            print(f"Best Sample PPL: {metrics['best_sample_ppl']:.4f}")
            print(f"Best Sample PPL Improvement: {metrics['best_sample_improvement']:.4f}")
            print(f"Best Sample Semantic Similarity: {metrics['best_sample_semantic_sim']:.4f}")
    
    print(f"==========================\n")


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    # Determine initialization strategy (same logic as generation script)
    if "inv-first" in cfg.training.method:
        use_init = 'bos'
    elif "bert-like" in cfg.training.method or "pag-mix-identity-score-embeddings" in cfg.training.method:
        use_init = 'pad'
    elif "identity-grad" in cfg.training.method or cfg.training.method == "base":
        use_init = 'bigram'
    else:
        raise ValueError(f"Unsupported training method: {cfg.training.method}")

    # Validate checkpoint path
    raw_ckpt = cfg.model.checkpoint_path
    if raw_ckpt is None or str(raw_ckpt).strip().lower() in ("", "none"):
        raise ValueError(f"Model checkpoint path is not set or invalid: {raw_ckpt!r}")
    ckpt_path = Path(str(raw_ckpt))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found: {ckpt_path}")

    # Get model name for file pattern matching
    lightning_module, data_module, model_name, cfg = __import__('instantiate').load_model_from_checkpoint(
        ckpt_path, cfg
    )
    
    # Find all result files for this model and initialization
    results_pattern = f'backward_inference_{model_name}_{use_init}_*.json'
    output_dir = cfg.model.output_dir
    
    # Look for worker result files
    results_files = list(output_dir.glob(results_pattern))
    if not results_files:
        # Try without worker suffix (single GPU run)
        single_file = output_dir / f'backward_inference_{model_name}_{use_init}.json'
        if single_file.exists():
            results_files = [str(single_file)]
        else:
            raise FileNotFoundError(f"No backward inference result files found matching pattern: {results_pattern}")
    
    results_files = [str(f) for f in results_files]
    print(f"Found result files: {results_files}")
    
    # Load all results
    all_results = load_backward_inference_results(results_files)
    if not all_results:
        raise ValueError("No results loaded from files")
    
    # Setup WandB logger for evaluation
    run_name = f"backward-eval-{cfg.training.method}-{use_init}"
    tags = ["backward-eval", cfg.training.method, cfg.dataset.name, use_init]
    if cfg.training.method == "pag-hidden":
        run_name += f"-{cfg.model.hidden_layer_index}-classes-{cfg.training.pag_classes}"
        tags += [f"layer-{cfg.model.hidden_layer_index}", f"pag-classes-{cfg.training.pag_classes}"]

    wandb_logger = WandbLogger(
        entity='pag-llm-team',
        project='pag-llm-backward-inference',
        name=run_name,
        tags=tags,
        config={
            **dataclasses.asdict(cfg),
            'evaluation_params': {
                'num_results': len(all_results),
                'use_init': use_init,
                'model_name': model_name,
                'results_files': results_files,
                'batch_size': 32,
            }
        },
    )
    
    # Initialize models for evaluation
    print("Loading models for evaluation...")
    
    # Load main model
    lightning_module, _, _, _, _ = init_evaluation(
        cfg=cfg,
        device='cuda:0',
        use_init=use_init,
        ckpt_file=str(ckpt_path),
    )
    lightning_module.eval()
    
    # Load baseline model if needed
    baseline_model = None
    if use_init == 'bigram':
        baseline_ckpt_path = ckpt_path.parent / 'best-base.ckpt'
        if baseline_ckpt_path.exists():
            try:
                baseline_model, _, _, _, _ = init_evaluation(
                    cfg=cfg,
                    device='cuda:0',
                    use_init=use_init,
                    ckpt_file=str(baseline_ckpt_path),
                )
                baseline_model.eval()
                print("Loaded baseline model for comparison")
            except Exception as e:
                print(f"Warning: Could not load baseline model: {e}")
        else:
            print("Warning: Baseline model not found for bigram comparison")
    
    # Load semantic similarity model
    print("Loading SentenceTransformer model for semantic similarity...")
    semantic_model = load_semantic_model(cfg)
    
    # Initialize evaluator
    external_llm_path = str(cfg.model.local_external_llm_path) or cfg.model.external_llm or "gpt2"
    evaluator = BackwardInferenceEvaluator(forward_model=lightning_module, external_llm=external_llm_path)
    
    # Compute metrics
    batch_size = 32  # Adjust based on available VRAM
    sample_metrics, aggregate_metrics = compute_backward_inference_metrics(
        results=all_results,
        lightning_model=lightning_module,
        baseline_model=baseline_model,
        semantic_model=semantic_model,
        evaluator=evaluator,
        batch_size=batch_size,
        output_dir=output_dir,
        model_name=model_name,
        use_init=use_init
    )
    
    # Log metrics to WandB
    wandb_logger.experiment.log(aggregate_metrics)
    
    # Log best samples to WandB if available
    if 'num_best_samples_saved' in aggregate_metrics and aggregate_metrics['num_best_samples_saved'] > 0:
        log_best_samples_to_wandb(wandb_logger, output_dir, model_name, use_init)
    
    # Print summary
    comprehensive_metrics = {k: v for k, v in aggregate_metrics.items() 
                           if k.startswith('mean_')}
    print_evaluation_summary(aggregate_metrics, comprehensive_metrics)
    
    # Finish WandB run
    wandb_logger.experiment.finish()
    
    print(f"Optimized evaluation completed for {len(all_results)} samples")


if __name__ == '__main__':
    main()  # type: ignore
