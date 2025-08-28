import dataclasses
import json
import pathlib
import os
from datetime import datetime
import torch.multiprocessing as mp

import torch
import wandb

from config import CustomLLMPagConfig, apply_config
from gcg import gcg_algorithm, gcg_evaluation
from gcg.gcg_utils import get_gpu_count
from instantiate import load_model_from_checkpoint


def run_gcg_worker(
        rank: int, world_size: int, cfg: CustomLLMPagConfig,
        model_checkpoint_path: str | pathlib.Path,
        all_sample_indices: list[int],
        num_prefix_tokens: int,
        num_steps: int,
        search_width: int,
        top_k: int,
        output_dir: str | pathlib.Path,
    ):
    """
    Worker process for running GCG on a subset of samples.
    """
    # Set up device
    # Map rank to GPU index. Ensure we pass an integer to set_device to avoid
    # ambiguous behavior across torch versions and with spawn method.
    gpu_index = int(rank)
    device = f'cuda:{gpu_index}'
    try:
        torch.cuda.set_device(gpu_index)
    except Exception as e:
        # Print and re-raise so the parent sees the traceback in logs
        print(f"Worker {rank}: failed to set CUDA device {gpu_index}: {e}", flush=True)
        raise
    torch.set_float32_matmul_precision('medium')

    # Print some diagnostic info so logs show why a worker might hang.
    try:
        cuda_count = torch.cuda.device_count()
    except Exception:
        cuda_count = 'unknown'
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')
    print(f"Worker {rank}: starting on device {device}. PID={os.getpid()}; torch.cuda.device_count()={cuda_count}; CUDA_VISIBLE_DEVICES={cuda_visible}", flush=True)

    # Load model and data
    path_str = None if model_checkpoint_path is None else str(model_checkpoint_path)
    if not path_str or path_str == 'None':
        raise ValueError(f"Worker {rank}: model_checkpoint_path is not set or is invalid: {model_checkpoint_path!r}")
    ckpt_path = pathlib.Path(path_str)
    try:
        lightning_model, data_module, _, _ = load_model_from_checkpoint(
            ckpt_path,
            cfg,
        )
    except Exception as e:
        print(f"Worker {rank}: error loading checkpoint {ckpt_path}: {e}", flush=True)
        raise
    if data_module.test_dataset is None:
        raise ValueError("Test dataset is not available in the data module.")
    lightning_model.to(device).eval()

    # Create GCG instance
    gcg = gcg_algorithm.GCG(
        model=lightning_model.model,
        tokenizer=lightning_model.tokenizer,
        num_prefix_tokens=num_prefix_tokens,
        num_steps=num_steps,
        search_width=search_width,
        top_k=top_k,
    )

    # Get subset of samples for this worker
    samples_for_worker = all_sample_indices[rank::world_size]
    print(f"Worker {rank}: attacking {len(samples_for_worker)} samples (indices sample).", flush=True)
    try:
        # Run evaluation
        worker_results = gcg_evaluation.evaluate_model_with_gcg(
            gcg,
            data_module.test_dataset,
            target_response_len=10,
            samples_to_attack=samples_for_worker,
            process_rank=rank
        )

        # Write per-worker results to a file to avoid Queue pipe deadlocks
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'worker_{rank}.json'
        with out_file.open('w') as f:
            json.dump([r.to_dict() for r in worker_results], f)
        print(f"Worker {rank}: wrote results to {out_file}", flush=True)
    except Exception as e:
        # Catch exceptions so worker prints an error and doesn't silently hang
        print(f"Worker {rank}: exception during evaluation: {e}", flush=True)
        # Write a single unified worker file containing the error message
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'worker_{rank}.json'
        with out_file.open('w') as f:
            json.dump({"error": str(e)}, f)
        # Re-raise so logs show the traceback if spawn prints it
        raise


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    """
    Main function to train the model with the Inverse First Token task.

    Args:
        cfg: Configuration object with all parameters
    """
    # Instantiate model and data module
    # Robustly validate checkpoint path: reject None, empty string, or the literal string 'None'
    raw_ckpt = cfg.model.checkpoint_path
    if raw_ckpt is None or str(raw_ckpt).strip().lower() in ("", "none"):
        raise ValueError(f"Model checkpoint path is not set or invalid: {raw_ckpt!r}")
    ckpt_path = pathlib.Path(str(raw_ckpt))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found: {ckpt_path}")

    lightning_model, data_module, model_name, cfg = load_model_from_checkpoint(
        ckpt_path,
        cfg,
    )
    if data_module.test_dataset is None:
        raise ValueError("Test dataset is not available in the data module.")

    # GCG parameters
    num_prefix_tokens=20 # GCG original work uses 20
    num_steps = 500
    search_width=4096 # GCG original work uses 512 as "batch size"
    top_k=256 # GCG original work uses 256

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
    })

    # Determine number of GPUs and set up multiprocessing
    world_size = get_gpu_count(cfg)
    print(f"Found {world_size} GPUs. Starting parallel GCG evaluation.")
    wandb.log({"model_info/device": f"{world_size} GPUs"})

    # Determine samples to attack
    dataset = data_module.test_dataset
    max_samples_to_attack = int(len(dataset) * 0.1)
    all_sample_indices = torch.randperm(len(dataset))[:max_samples_to_attack].tolist()

    # Use 'spawn' start method for CUDA compatibility in multiprocessing
    mp.set_start_method('spawn', force=True)

    # Prepare temporary per-worker output directory
    tmp_out_dir = cfg.model.output_dir / f'gcg_{model_name}_tmp'
    if not tmp_out_dir.exists():
        tmp_out_dir.mkdir(parents=True, exist_ok=True)

    # Spawn worker processes
    print(f"Spawning workers with checkpoint: {ckpt_path}")
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_gcg_worker,
            # Pass checkpoint path as string to avoid pickling/pathlib issues in spawned processes
            args=(rank, world_size, cfg, str(ckpt_path), all_sample_indices,
                    num_prefix_tokens, num_steps, search_width, top_k, str(tmp_out_dir)),
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Collect results from per-worker files (each worker writes worker_<rank>.json)
    gcg_results_dicts = []
    files = sorted(pathlib.Path(tmp_out_dir).glob('worker_*.json'))
    if not files:
        print("No GCG result files found.")
        return
    for fpath in files:
        with fpath.open() as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Failed to read {fpath}: {e}")
                continue
            # Unified format: either a list of result dicts or a dict with an 'error' key
            if isinstance(data, list):
                gcg_results_dicts.extend(data)
            elif isinstance(data, dict) and data.get('error') is not None:
                print(f"Worker file {fpath} contains error: {data['error']}")
            else:
                gcg_results_dicts.append(data)

    # Save final results
    if not gcg_output_file.parent.exists():
        gcg_output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving GCG results to {gcg_output_file}")
    with gcg_output_file.open('w') as f:
        json.dump(gcg_results_dicts, f, indent=4)
    print(f"Saved GCG results to {gcg_output_file}")

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
