import dataclasses
import json
from pathlib import Path

import torch
from lightning.pytorch.loggers import WandbLogger
from sentence_transformers import SentenceTransformer

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


def compute_backward_inference_metrics(
        results: list[dict],
        lightning_model: BaseLMModel,
        baseline_model: BaseLMModel | None,
        semantic_model: SentenceTransformer,
        evaluator: BackwardInferenceEvaluator
) -> tuple[list[dict], dict]:
    """
    Compute comprehensive metrics for backward inference results.
    
    Args:
        results: List of backward inference results
        lightning_model: Main trained model
        baseline_model: Baseline model for comparison (optional)
        semantic_model: Sentence transformer for semantic similarity
        evaluator: Backward inference evaluator
        
    Returns:
        tuple: (sample_metrics, aggregate_metrics)
    """
    print("Computing comprehensive metrics for backward inference results...")
    
    comprehensive_sample_metrics = []
    total_predicted_ppl = 0.0
    total_original_ppl = 0.0
    total_bigram_ppl = 0.0 if baseline_model is not None else None
    total_semantic_similarity = 0.0
    total_bigram_semantic_similarity = 0.0 if baseline_model is not None else None
    
    for i, result in enumerate(results):
        try:
            # Convert lists back to tensors
            original_input_ids = torch.tensor(result['original_input_ids'], dtype=torch.long)
            original_attention_mask = torch.tensor(result['original_attention_mask'], dtype=torch.long)
            predicted_input_ids = torch.tensor(result['predicted_input_ids'], dtype=torch.long)
            predicted_attention_mask = torch.tensor(result['predicted_attention_mask'], dtype=torch.long)
            
            # Move to device
            device = lightning_model.device
            original_input_ids = original_input_ids.to(device)
            original_attention_mask = original_attention_mask.to(device)
            predicted_input_ids = predicted_input_ids.to(device)
            predicted_attention_mask = predicted_attention_mask.to(device)
            
            # Compute perplexities
            predicted_ppl = get_batch_perplexity(
                lightning_model, 
                predicted_input_ids.unsqueeze(0), 
                predicted_attention_mask.unsqueeze(0)
            ).item()
            
            original_ppl = get_batch_perplexity(
                lightning_model,
                original_input_ids.unsqueeze(0),
                original_attention_mask.unsqueeze(0)
            ).item()
            
            # Compute semantic similarity
            original_text = result['original_text']
            predicted_text = result['predicted_text']
            semantic_sim = compute_semantic_similarity(original_text, predicted_text, semantic_model)
            
            # Accumulate totals
            total_predicted_ppl += predicted_ppl
            total_original_ppl += original_ppl
            total_semantic_similarity += semantic_sim
            
            # Compute comprehensive metrics using evaluator
            prefix_len = result['prefix_len']
            original_prefix_ids = original_input_ids[:prefix_len]
            predicted_prefix_ids = predicted_input_ids[:prefix_len]
            suffix_ids = torch.tensor(result['suffix_input_ids'], dtype=torch.long).to(device)
            
            # Create attention masks
            original_prefix_mask = original_attention_mask[:prefix_len]
            predicted_prefix_mask = predicted_attention_mask[:prefix_len]
            suffix_mask = torch.tensor(result['suffix_attention_mask'], dtype=torch.long).to(device)
            
            sample_metrics = evaluator.compute_comprehensive_metrics(
                reference_prefix=original_prefix_ids,
                generated_prefix=predicted_prefix_ids,
                true_suffix=suffix_ids,
                reference_prefix_mask=original_prefix_mask,
                generated_prefix_mask=predicted_prefix_mask,
                suffix_mask=suffix_mask,
                predicted_overall_text=predicted_text,
                original_overall_text=original_text
            )
            comprehensive_sample_metrics.append(sample_metrics)
            
            # Baseline comparison if available
            if baseline_model is not None and result['use_init'] == 'bigram':
                # For bigram baseline, we would need to generate baseline predictions
                # For now, we'll skip this complex comparison
                pass
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(results)} samples")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Compute aggregate metrics
    actual_samples = len(results)
    avg_predicted_ppl = total_predicted_ppl / actual_samples
    avg_original_ppl = total_original_ppl / actual_samples
    avg_semantic_similarity = total_semantic_similarity / actual_samples
    
    # Aggregate comprehensive metrics
    aggregated_comprehensive_metrics = aggregate_metrics(comprehensive_sample_metrics)
    
    aggregate_final_metrics = {
        'avg_predicted_overall_ppl': avg_predicted_ppl,
        'avg_original_overall_ppl': avg_original_ppl,
        'avg_semantic_similarity': avg_semantic_similarity,
        'ppl_improvement': avg_original_ppl - avg_predicted_ppl,
        'ppl_improvement_ratio': avg_predicted_ppl / avg_original_ppl if avg_original_ppl > 0 else float('inf'),
        'actual_samples_used': actual_samples,
    }
    
    # Add aggregated comprehensive metrics
    aggregate_final_metrics.update(aggregated_comprehensive_metrics)
    
    if total_bigram_ppl is not None:
        avg_bigram_ppl = total_bigram_ppl / actual_samples
        avg_bigram_semantic_similarity = (total_bigram_semantic_similarity or 0.0) / actual_samples
        aggregate_final_metrics.update({
            'avg_bigram_overall_ppl': avg_bigram_ppl,
            'avg_bigram_semantic_similarity': avg_bigram_semantic_similarity,
            'bigram_vs_predicted_ppl_ratio': avg_predicted_ppl / avg_bigram_ppl if avg_bigram_ppl > 0 else float('inf'),
            'semantic_similarity_improvement': avg_semantic_similarity - avg_bigram_semantic_similarity,
        })
    
    return comprehensive_sample_metrics, aggregate_final_metrics


def print_evaluation_summary(metrics: dict, comprehensive_metrics: dict):
    """Print a comprehensive summary of evaluation results."""
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Samples Used: {metrics['actual_samples_used']}")
    print(f"Average Predicted PPL: {metrics['avg_predicted_overall_ppl']:.2f}")
    print(f"Average Original PPL: {metrics['avg_original_overall_ppl']:.2f}")
    print(f"Average Semantic Similarity: {metrics['avg_semantic_similarity']:.4f}")
    print(f"PPL Improvement: {metrics['ppl_improvement']:.2f}")
    print(f"PPL Improvement Ratio: {metrics['ppl_improvement_ratio']:.3f}")
    
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
    print(f"Mean 3rd-Party Original txt PPL: {comprehensive_metrics.get('original_perplexity', float('inf')):.2f}")
    print(f"Mean 3rd-Party Predicted txt PPL: {comprehensive_metrics.get('predicted_perplexity', float('inf')):.2f}")

    if 'avg_bigram_overall_ppl' in metrics:
        print(f"\n--- Bigram Baseline Comparison ---")
        print(f"Average Bigram PPL: {metrics['avg_bigram_overall_ppl']:.2f}")
        print(f"Average Bigram Semantic Similarity: {metrics['avg_bigram_semantic_similarity']:.4f}")
        print(f"Bigram vs Predicted PPL Ratio: {metrics['bigram_vs_predicted_ppl_ratio']:.3f}")
        print(f"Semantic Similarity Improvement: {metrics['semantic_similarity_improvement']:.4f}")
    
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
    sample_metrics, aggregate_metrics = compute_backward_inference_metrics(
        results=all_results,
        lightning_model=lightning_module,
        baseline_model=baseline_model,
        semantic_model=semantic_model,
        evaluator=evaluator
    )
    
    # Log metrics to WandB
    wandb_logger.experiment.log(aggregate_metrics)
    
    # Print summary
    comprehensive_metrics = {k: v for k, v in aggregate_metrics.items() 
                           if k.startswith('mean_')}
    print_evaluation_summary(aggregate_metrics, comprehensive_metrics)
    
    # Finish WandB run
    wandb_logger.experiment.finish()
    
    print(f"Evaluation completed for {len(all_results)} samples")


if __name__ == '__main__':
    main()  # type: ignore
