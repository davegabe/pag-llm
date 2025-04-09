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
from config import LLMPagConfig, apply_config
from data.data_module import LMDataModule
from data.data_processor import BatchType
from utils.hdf5 import save_hidden_states_to_hdf5


def extract_prefixes_and_next_tokens(
        batch: BatchType,
    max_seq_length: int,
    prefix_max_length: int,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract prefixes and their corresponding next tokens from input sequences.

    Args:
        batch: Batched text data from TextDataset
        max_seq_length: Maximum sequence length
        prefix_max_length: Maximum prefix length
        device: Torch device

    Returns:
        prefixes: Tensor of prefix token IDs [batch_size, prefix_len]
        next_tokens: Tensor of next token IDs [batch_size, prefix_len]
    """
    batch_size = batch.input_ids.size(0)

    # Get the index of the start token of each sentence (ignoring padding)
    sentences_start_idx: torch.Tensor = max_seq_length - batch.attention_mask.sum(dim=1)
    
    # To get the prefix tokens using batched operations:
    # - Start with the index of the start token of each sentence (sentences_start_idx)
    # - Add the shift per token requested, the same for every sentence
    # - Add the shift to have a flattened index to pass to torch.take()
    flat_start_ids = sentences_start_idx[:, None] + \
                        torch.arange(prefix_max_length, device=device).repeat(batch_size).view(batch_size, -1) + \
                        torch.arange(batch_size, device=device)[:, None] * max_seq_length

    # Extract the prefixes from the input_ids
    prefixes = torch.take(batch.input_ids, flat_start_ids)

    # Double check that the attention masks are all ones
    attn_mask_items = torch.take(batch.attention_mask, flat_start_ids).sum().cpu().item()
    prefixes_items_count = reduce(lambda x, y: x * y, prefixes.shape, 1)
    assert attn_mask_items == prefixes_items_count

    # Get the next token = same indexes as prefixes, but in the labels Tensor
    next_tokens = torch.take(batch.labels, flat_start_ids)

    return prefixes, next_tokens


def process_batch(
    model: torch.nn.Module,
    pad_token_id: int,
        batch: BatchType,
    device: torch.device,
        cfg: LLMPagConfig,
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
    # Extract prefixes and next tokens
    prefixes, next_tokens = extract_prefixes_and_next_tokens(
        batch=batch,
        max_seq_length=cfg.training.max_seq_length,
        prefix_max_length=cfg.dataset.prefix.max_length,
        device=device,
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
    prefix_hidden_state = selected_layer_hidden_state[:, cfg.dataset.prefix.min_length:]

    # As well, ignore the first min_length tokens, to make them in sync with the prefix_hidden_state
    prefix_next_tokens = next_tokens[:, cfg.dataset.prefix.min_length:]

    # Save the hidden states and next tokens
    samples_saved = save_hidden_states_to_hdf5(
        hidden_states=prefix_hidden_state,
        next_tokens=prefix_next_tokens,
        file_path=output_file,
        append=current_batch_idx > 0
    )

    return samples_saved


@apply_config()
@torch.no_grad()
def main(cfg: LLMPagConfig):
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
        cfg.model.random_initialization,
        lora_config=cfg.training.lora,
    )
    model.to(device)
    model.eval()

    # Load the training dataset
    data_module = LMDataModule(cfg, tokenizer)
    data_module.prepare_data()
    data_module.setup()
    train_dataloader = data_module.train_dataloader()

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
            batch: BatchType = batch.to(device)
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

            total_processed += batch.input_ids.size(0)

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
