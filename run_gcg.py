import json
import os
import pathlib

import torch

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
        top_k: int
) -> list[dict]:
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
    # noinspection PyBroadException
    try:
        cuda_count = torch.cuda.device_count()
    except:
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

        # Put results in the queue
        return [r.to_dict() for r in worker_results]
    except Exception as e:
        # Catch exceptions so worker prints an error and doesn't silently hang
        print(f"Worker {rank}: exception during evaluation: {e}", flush=True)
        raise


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    print(cfg, cfg.training, cfg.training.gpu_rank, sep='\n', end='\n\n')
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
    multi_gpu_suffix = '' if cfg.training.gpu_rank is None else f'_{cfg.training.gpu_rank}'
    gcg_output_file = cfg.model.output_dir / f'gcg_{model_name}{multi_gpu_suffix}.json'
    if gcg_output_file.exists():
        print(f"File {gcg_output_file} already exists. Skipping GCG evaluation.")
        return
    print(f"GCG results will be saved to {gcg_output_file}")

    # Determine number of GPUs and set up multiprocessing
    world_size = get_gpu_count(cfg)
    print(f"Found {world_size} GPUs. Starting parallel GCG evaluation.")

    # Determine samples to attack
    dataset = data_module.test_dataset
    max_samples_to_attack = int(len(dataset) * 0.1)
    torch.manual_seed(0)
    all_sample_indices = torch.randperm(len(dataset))[:max_samples_to_attack].tolist()
    print(f'Worker {cfg.training.gpu_rank}: we include sample no. {all_sample_indices[0]}.')

    # Spawn worker processes
    print(f"Running worker {cfg.training.gpu_rank} with checkpoint: {ckpt_path}")
    gcg_results_dicts = run_gcg_worker(
        rank=cfg.training.gpu_rank if cfg.training.gpu_rank is not None else 0,
        world_size=world_size,
        cfg=cfg,
        model_checkpoint_path=ckpt_path,
        all_sample_indices=all_sample_indices,
        num_prefix_tokens=num_prefix_tokens,
        num_steps=num_steps,
        search_width=search_width,
        top_k=top_k,
    )

    # Save final results
    if not gcg_output_file.parent.exists():
        gcg_output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving GCG results to {gcg_output_file}")
    with gcg_output_file.open('w') as f:
        json.dump(gcg_results_dicts, f, indent=4)
    print(f"Saved GCG results to {gcg_output_file}")


if __name__ == '__main__':
    main() # type: ignore
