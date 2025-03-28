"""
File to generate a precomputed dataset of hidden states.
"""
import os
from functools import reduce

import torch
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput

import models.loader as loader
from config import Config, apply_config
from data.data_processor import load_and_process_dataloader
from utils.hdf5 import save_hidden_states_to_hdf5


def extract_prefixes_and_next_tokens(
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    max_seq_length: int,
    prefix_max_length: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract prefixes and their corresponding next tokens from input sequences.

    Args:
        input_ids: Input token IDs [batch_size, seq_length]
        attn_mask: Attention mask [batch_size, seq_length]
        max_seq_length: Maximum sequence length
        prefix_max_length: Maximum prefix length
        device: Torch device

    Returns:
        prefixes: Tensor of prefix token IDs [batch_size, prefix_len]
        next_tokens: Tensor of next token IDs [batch_size, 1]
    """
    batch_size = input_ids.size(0)

    # Get the index of the start token of each sentence (ignoring padding)
    sentences_start_idx: torch.Tensor = max_seq_length - attn_mask.sum(dim=1)
    
    # To get the prefix tokens using batched operations:
    # - Start with the index of the start token of each sentence (sentences_start_idx)
    # - Add the shift per token requested, the same for every sentence
    # - Add the shift to have a flattened index to pass to torch.take()
    flat_start_ids = sentences_start_idx[:, None] + \
                        torch.arange(prefix_max_length, device=device).repeat(batch_size).view(batch_size, -1) + \
                        torch.arange(batch_size, device=device)[:, None] * max_seq_length

    # Extract the prefixes from the input_ids
    prefixes = torch.take(input_ids, flat_start_ids)

    # Double check that the attention masks are all ones
    attn_mask_items = torch.take(attn_mask, flat_start_ids).sum().cpu().item()
    prefixes_items_count = reduce(lambda x, y: x * y, prefixes.shape, 1)
    assert attn_mask_items == prefixes_items_count

    # Get the next token after the last position in the prefix
    next_token_indices = flat_start_ids[:, -1] + 1
    next_tokens = torch.take(input_ids, next_token_indices).unsqueeze(1)

    return prefixes, next_tokens


def process_batch(
    model: torch.nn.Module,
    pad_token_id: int,
    batch: dict,
    device: torch.device,
    cfg: Config,
    output_file: str,
    current_batch_idx: int
) -> int:
    """
    Process a single batch to generate and save hidden states.

    Args:
        model: The language model
        pad_token_id: ID of the padding token
        batch: Dictionary containing batch data
        device: Torch device
        cfg: Configuration object
        output_file: Path to output HDF5 file
        current_batch_idx: Current batch index

    Returns:
        int: Number of samples saved to the file
    """
    # Move inputs to the device
    input_ids = batch['input_ids'].to(device)
    attn_mask = batch['attention_mask'].to(device)

    # Extract prefixes and next tokens
    prefixes, next_tokens = extract_prefixes_and_next_tokens(
        input_ids=input_ids,
        attn_mask=attn_mask,
        max_seq_length=cfg.training.max_seq_length,
        prefix_max_length=cfg.dataset.prefix.max_length,
        device=device
    )

    # Generate hidden states for the prefixes
    outputs: GenerateDecoderOnlyOutput = model.generate(
        input_ids=prefixes,
        attention_mask=torch.ones_like(prefixes),
        generation_config=GenerationConfig(
            max_new_tokens=1,  # We only need the hidden states for the prefix tokens
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id,
        ),
    )

    # Get the hidden states from the desired layer
    selected_layer_hidden_state = outputs.hidden_states[0][cfg.model.hidden_layer_index]

    # Now we have to ignore the first min_length tokens, since they are only context, not our desired X
    prefix_hidden_state = selected_layer_hidden_state[:, cfg.dataset.prefix.min_length:, :]

    # Save the hidden states and next tokens
    samples_saved = save_hidden_states_to_hdf5(
        hidden_states=prefix_hidden_state,
        next_tokens=next_tokens,
        file_path=output_file,
        append=current_batch_idx > 0
    )

    return samples_saved


@apply_config()
@torch.no_grad()
def main(cfg: Config):
    """
    Main function to generate and save hidden states from model prefixes.

    Args:
        cfg: Configuration object with all parameters
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate configuration
    if cfg.dataset.prefix.min_length >= cfg.dataset.prefix.max_length:
        raise ValueError("Prefix min_length must be less than max_length")

    # Load model and tokenizer
    model, tokenizer = loader.load_model_and_tokenizer(
        cfg.model.pretrained_base,
        use_lora=cfg.training.use_lora
    )
    model.to(device)
    model.eval()

    # Load the training dataset
    train_dataloader, _ = load_and_process_dataloader(
        cfg.dataset,
        cfg.training,
        tokenizer,
        cfg.training.max_seq_length
    )

    # Create the output directory if it doesn't exist
    output_dir = cfg.model.output_dir
    os.makedirs(output_dir, exist_ok=True)
    hidden_layer_idx = cfg.model.hidden_layer_index
    output_file = os.path.join(output_dir, f"hidden_states_layer{hidden_layer_idx}.hdf5")

    # Process batches
    total_processed = 0
    start_batch = 0
    samples_saved = 0

    batch_idx = 0  # Define it anyway

    try:
        # Skip batches if resuming
        dataloader_iter = iter(train_dataloader)
        for _ in range(start_batch):
            next(dataloader_iter)

        # Process remaining batches
        pbar = tqdm(dataloader_iter, desc="Processing batches")
        for batch_idx, batch in enumerate(pbar):
            batch: dict
            current_batch_idx = batch_idx + start_batch

            # Process the current batch
            samples_saved = process_batch(
                model=model,
                pad_token_id=tokenizer.pad_token_id,
                batch=batch,
                device=device,
                cfg=cfg,
                output_file=output_file,
                current_batch_idx=current_batch_idx
            )

            total_processed += batch['input_ids'].size(0)

            # Write summary to the progress bar
            pbar.set_postfix(samples_saved=samples_saved)

            # Free up memory
            torch.cuda.empty_cache()

        print(f"Finished processing {total_processed} samples. (saved {samples_saved} samples).")

    except KeyboardInterrupt:
        print("Interrupted by user. Saving the current progress...")
        print(f"Total samples processed: {total_processed}")
        print(f"Last batch index: {batch_idx + start_batch}")
        print(f"Total samples saved: {samples_saved}")
    except Exception as e:
        print(f"Error occurred during processing: {str(e)} ")
        raise


if __name__ == "__main__":
    main()
