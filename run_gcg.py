import dataclasses
import json
import pathlib
import time
from datetime import datetime

import torch
import wandb

from config import CustomLLMPagConfig, apply_config
from data.data_processor import PreTokenizedDataset, TextDataset
from gcg import gcg_algorithm, gcg_evaluation
from instantiate import load_model_from_checkpoint


def run_gcg(
        gcg: gcg_algorithm.GCG,
        dataset: TextDataset | PreTokenizedDataset,
        gcg_output_file: pathlib.Path, 
    ):
    """
    Run GCG evaluation on a dataset and log results.

    Args:
        gcg: GCG algorithm instance
        dataset: Dataset to evaluate on
        gcg_output_file: Path to save final results
        lightning_model: Language model for computing naturalness metrics
        cfg: Configuration object for loading semantic model
    """
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
    
    # Run single evaluation with intermediate logging
    gcg_results = gcg_evaluation.evaluate_model_with_gcg(
        gcg, 
        dataset,
        target_response_len=10,
        max_samples_to_attack=int(len(dataset) * 0.1),  # Attack 10% of the dataset
        random_select_samples=True
    )
    
    evaluation_time = time.time() - start_time
    
    # Log evaluation time and basic stats
    wandb.log({
        "evaluation/total_time": evaluation_time,
        "evaluation/samples_attacked": len(gcg_results),
        "evaluation/avg_time_per_sample": evaluation_time / len(gcg_results) if gcg_results else 0
    })
    
    # Save final results
    if not gcg_output_file.parent.exists():
        gcg_output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving GCG results to {gcg_output_file}")
    with gcg_output_file.open('w') as f:
        json.dump([r.to_dict() for r in gcg_results], f, indent=4)
    print(f"Saved GCG results to {gcg_output_file}")
    
    return gcg_results

@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    """
    Main function to train the model with the Inverse First Token task.

    Args:
        cfg: Configuration object with all parameters
    """
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Instantiate model and data module
    if cfg.model.checkpoint_path is None:
        raise ValueError("Model checkpoint path is not set.")
    lightning_model, data_module, model_name, cfg = load_model_from_checkpoint(
        cfg.model.checkpoint_path,
        cfg,
    )
    if data_module.test_dataset is None:
        raise ValueError("Test dataset is not available in the data module.")

    # Set device for model
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.training.device is not None:
        torch_device = f'cuda:{cfg.training.device[0]}'
    lightning_model.to(torch_device).eval()

    # Run GCG
    num_steps = 500
    gcg = gcg_algorithm.GCG(
        model=lightning_model.model,
        tokenizer=lightning_model.tokenizer,
        num_prefix_tokens=20, # GCG original work uses 20
        num_steps=num_steps, # Extended to num_steps for convergence analysis
        search_width=4096, # GCG original work uses 512 as "batch size"
        top_k=256, # GCG original work uses 256
    )

    # Check if output file already exists
    gcg_output_file = cfg.model.output_dir / f'gcg_{model_name}.json'
    if gcg_output_file.exists():
        print(f"File {gcg_output_file} already exists. Skipping GCG evaluation.")
        return

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
    
    # Log model information
    total_params = sum(p.numel() for p in lightning_model.parameters())
    trainable_params = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)
    wandb.log({
        "model_info/total_parameters": total_params,
        "model_info/trainable_parameters": trainable_params,
        "model_info/model_name": model_name,
        "model_info/device": str(torch_device)
    })

    # Run GCG and save results
    run_gcg(
        gcg, 
        data_module.test_dataset,
        gcg_output_file
    )
    
    # Log the output file as an artifact
    artifact = wandb.Artifact(
        name=f"gcg_results_{model_name}",
        type="results",
        description=f"GCG attack results for model {model_name} with convergence analysis"
    )
    artifact.add_file(str(gcg_output_file))
    wandb.log_artifact(artifact)

    if wandb.run is not None:
        print(f"Experiment logged to wandb: {wandb.run.url}")
    wandb.finish()


if __name__ == '__main__':
    main() # type: ignore
