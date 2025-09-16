import json
import os
import pathlib
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from config import CustomLLMPagConfig, apply_config
from gcg.gcg_utils import get_gpu_count
from instantiate import load_model_from_checkpoint
from inverse_lm_stats import init_evaluation
from models.base_model import BaseLMModel


def determine_use_init(cfg: CustomLLMPagConfig) -> str:
    """
    Determine the initialization strategy based on the training method.
    
    Args:
        cfg: Configuration object
        
    Returns:
        str: Initialization strategy ('bos', 'pad', 'bigram')
    """
    if "inv-first" in cfg.training.method:
        print(f"Method {cfg.training.method} need to use BOS for initialization")
        return 'bos'
    elif "bert-like" in cfg.training.method or "pag-mix-identity-score-embeddings" in cfg.training.method:
        print(f"Method {cfg.training.method} need to use PAD for initialization")
        return 'pad'
    elif "identity-grad" in cfg.training.method or cfg.training.method == "base":
        print(f"Method {cfg.training.method} need to use PAG for initialization")
        return 'bigram'
    else:
        raise ValueError(f"Unsupported training method: {cfg.training.method}")


def backward_inference_worker(
        rank: int, 
        world_size: int, 
        cfg: CustomLLMPagConfig,
        model_checkpoint_path: str | pathlib.Path,
        all_sample_indices: list[int],
        prefix_len: int,
        beam_size: int,
        skip_prefix_tokens: int,
        use_init: str
) -> list[dict]:
    """
    Worker process for running backward inference on a subset of samples.
    
    Args:
        rank: Worker rank (GPU index)
        world_size: Total number of workers
        cfg: Configuration object
        model_checkpoint_path: Path to model checkpoint
        all_sample_indices: List of all sample indices to process
        prefix_len: Length of prefix to generate
        beam_size: Beam size for inference
        skip_prefix_tokens: Number of prefix tokens to skip
        use_init: Initialization strategy
        
    Returns:
        list[dict]: List of prediction results for this worker's samples
    """
    # Set up device
    gpu_index = min(int(rank), torch.cuda.device_count() - 1)
    device = f'cuda:{gpu_index}'
    try:
        torch.cuda.set_device(gpu_index)
    except Exception as e:
        print(f"Worker {rank}: failed to set CUDA device {gpu_index}: {e}", flush=True)
        raise
    torch.set_float32_matmul_precision('medium')

    # Print diagnostic info
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
        lightning_module, _, data_module, reverse_bigram, bigram_counts = init_evaluation(
            cfg=cfg,
            device=device,
            use_init=use_init,
            ckpt_file=str(ckpt_path),
        )
    except Exception as e:
        print(f"Worker {rank}: error loading checkpoint {ckpt_path}: {e}", flush=True)
        raise
    
    if data_module.test_dataset is None:
        raise ValueError("Test dataset is not available in the data module.")
    
    lightning_module.to(device).eval()

    # Get subset of samples for this worker
    samples_for_worker = all_sample_indices[rank::world_size]
    print(f"Worker {rank}: processing {len(samples_for_worker)} samples", flush=True)
    
    # Import prediction functions from original script
    from infer_backward_tinystories import (
        backward_infer_prefix, 
        backward_infer_bigram_only,
        pretty_decode_tokens
    )
    
    worker_results = []
    
    # Create progress bar for this worker
    pbar = tqdm(
        total=len(samples_for_worker),
        desc=f"Worker {rank}",
        position=rank,
        leave=True,
        unit="samples"
    )
    
    try:
        # Process samples for this worker
        processed_samples = 0
        
        for batch_idx, batch in enumerate(data_module.test_dataloader()):
            batch = batch.to(torch.device(device))
            input_ids, attention_mask, labels, shift_labels = batch
            batch_size = input_ids.size(0)
            
            # Calculate global sample indices for this batch
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = batch_start_idx + batch_size
            
            # Check if any samples in this batch are assigned to this worker
            batch_sample_indices = list(range(batch_start_idx, batch_end_idx))
            worker_batch_samples = [idx for idx in batch_sample_indices if idx in samples_for_worker]
            
            if not worker_batch_samples:
                continue  # Skip this batch if no samples assigned to this worker
            
            # Remove prefix tokens
            input_ids = input_ids[:, skip_prefix_tokens:]
            attention_mask = attention_mask[:, skip_prefix_tokens:]
            t = input_ids.size(-1)
            
            for local_sample_idx, (sample_input_ids, sample_attention_mask) in enumerate(zip(input_ids, attention_mask)):
                global_sample_idx = batch_start_idx + local_sample_idx
                
                # Skip if this sample is not assigned to this worker
                if global_sample_idx not in samples_for_worker:
                    continue
                
                # Extract suffix (skip prefix_len tokens from the beginning)
                suffix_length = t - prefix_len
                if suffix_length <= 0:
                    pbar.write(f"Worker {rank}: Sample {global_sample_idx} too short, skipping")
                    pbar.update(1)
                    continue
                
                suffix_input_ids = sample_input_ids[prefix_len:t]
                suffix_attention_mask = sample_attention_mask[prefix_len:t]
                original_input_ids = sample_input_ids[:t]
                original_attention_mask = sample_attention_mask[:t]
                
                # Get original text
                original_text = pretty_decode_tokens(lightning_module.tokenizer, original_input_ids)
                
                try:
                    # Perform backward inference
                    if use_init == 'bigram':
                        predicted_prefix_ids = backward_infer_bigram_only(
                            bigram_counts=bigram_counts,
                            lightning_module=lightning_module,
                            suffix_input_ids=suffix_input_ids.unsqueeze(0),
                            suffix_attention_mask=suffix_attention_mask.unsqueeze(0),
                            prefix_tokens_len=prefix_len,
                            beam_size=beam_size
                        )
                    else:
                        predicted_input_ids, predicted_attention_mask = backward_infer_prefix(
                            lightning_module=lightning_module,
                            use_init=use_init,
                            reverse_bigram=reverse_bigram,
                            suffix_input_ids=suffix_input_ids.unsqueeze(0),
                            suffix_attention_mask=suffix_attention_mask.unsqueeze(0),
                            suffix_length=suffix_length,
                            beam_size=beam_size
                        )
                        predicted_prefix_ids = predicted_input_ids[0][:prefix_len]
                    
                    # Combine predicted prefix with original suffix to get full prediction
                    full_predicted_ids = torch.cat([predicted_prefix_ids, suffix_input_ids])
                    full_predicted_mask = torch.cat([
                        torch.ones_like(predicted_prefix_ids), 
                        suffix_attention_mask
                    ])
                    
                    # Get predicted text
                    predicted_text = pretty_decode_tokens(lightning_module.tokenizer, full_predicted_ids)
                    
                    # Store result
                    result = {
                        'sample_idx': global_sample_idx,
                        'original_input_ids': original_input_ids.cpu().tolist(),
                        'original_attention_mask': original_attention_mask.cpu().tolist(),
                        'predicted_input_ids': full_predicted_ids.cpu().tolist(),
                        'predicted_attention_mask': full_predicted_mask.cpu().tolist(),
                        'predicted_prefix_ids': predicted_prefix_ids.cpu().tolist(),
                        'suffix_input_ids': suffix_input_ids.cpu().tolist(),
                        'suffix_attention_mask': suffix_attention_mask.cpu().tolist(),
                        'original_text': original_text,
                        'predicted_text': predicted_text,
                        'prefix_len': prefix_len,
                        'suffix_length': suffix_length,
                        'use_init': use_init,
                        'beam_size': beam_size,
                    }
                    
                    worker_results.append(result)
                    processed_samples += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'processed': processed_samples,
                        'rate': f"{processed_samples/(pbar.format_dict.get('elapsed', 1)):.1f} samples/s" if pbar.format_dict.get('elapsed', 0) > 0 else "0.0 samples/s"
                    })
                        
                except Exception as e:
                    pbar.write(f"Worker {rank}: Error processing sample {global_sample_idx}: {e}")
                    pbar.update(1)
                    continue
        
        pbar.close()
        print(f"Worker {rank}: completed processing {processed_samples} samples", flush=True)
        return worker_results
        
    except Exception as e:
        pbar.close()
        print(f"Worker {rank}: exception during evaluation: {e}", flush=True)
        raise


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    print(cfg, cfg.training, cfg.training.gpu_rank, sep='\n', end='\n\n')
    
    # Determine initialization strategy
    use_init = determine_use_init(cfg)
    
    # Validate checkpoint path
    raw_ckpt = cfg.model.checkpoint_path
    if raw_ckpt is None or str(raw_ckpt).strip().lower() in ("", "none"):
        raise ValueError(f"Model checkpoint path is not set or invalid: {raw_ckpt!r}")
    ckpt_path = pathlib.Path(str(raw_ckpt))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found: {ckpt_path}")

    # Load model briefly to get model name and validate
    lightning_module, data_module, model_name, cfg = load_model_from_checkpoint(
        ckpt_path,
        cfg,
    )
    if data_module.test_dataset is None:
        raise ValueError("Test dataset is not available in the data module.")

    # Backward inference parameters
    prefix_len = 20  # How many tokens to predict
    beam_size = 5
    skip_prefix_tokens = 5  # How many tokens to skip entirely

    # Check if output file already exists
    multi_gpu_suffix = '' if cfg.training.gpu_rank is None else f'_{cfg.training.gpu_rank}'
    output_file = cfg.model.output_dir / f'backward_inference_{model_name}_{use_init}{multi_gpu_suffix}.json'
    if output_file.exists():
        print(f"File {output_file} already exists. Skipping backward inference generation.")
        return
    print(f"Backward inference results will be saved to {output_file}")

    # Determine number of GPUs and set up multiprocessing
    world_size = get_gpu_count(cfg)
    print(f"Found {world_size} GPUs. Starting parallel backward inference.")

    # Determine samples to process
    dataset = data_module.test_dataset
    max_samples_to_process = len(dataset)  # Process the entire test set
    torch.manual_seed(0)
    all_sample_indices = list(range(len(dataset)))  # Use all samples in order
    print(f'Worker {cfg.training.gpu_rank}: Processing entire test dataset.')
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Total samples to process: {max_samples_to_process}")
    print(f"Samples per worker: {len(all_sample_indices[cfg.training.gpu_rank if cfg.training.gpu_rank is not None else 0::world_size])}")

    # Run worker
    print(f"Running worker {cfg.training.gpu_rank} with checkpoint: {ckpt_path}")
    results_dicts = backward_inference_worker(
        rank=cfg.training.gpu_rank if cfg.training.gpu_rank is not None else 0,
        world_size=world_size,
        cfg=cfg,
        model_checkpoint_path=ckpt_path,
        all_sample_indices=all_sample_indices,
        prefix_len=prefix_len,
        beam_size=beam_size,
        skip_prefix_tokens=skip_prefix_tokens,
        use_init=use_init
    )

    # Save results
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving backward inference results to {output_file}")
    with output_file.open('w') as f:
        json.dump(results_dicts, f, indent=4)
    print(f"Saved backward inference results to {output_file}")


if __name__ == '__main__':
    main()  # type: ignore
