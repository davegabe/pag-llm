import os
import tempfile
from dataclasses import dataclass
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizerFast
from huggingface_hub import create_repo, upload_folder

from create_dataset import train_new_tokenizer, login_to_hf, DataConfig

def preprocess_and_chunk_dataset_inv_first(dataset: Dataset, tokenizer: PreTrainedTokenizerFast, config: DataConfig) -> Dataset:
    """Preprocesses text and chunks the dataset for inv-first format."""
    print("Step 2: Preprocessing and chunking dataset for inv-first format...")
    
    # --- Pre-processing based on endoftext_handling ---
    def handle_endoftext(examples):
        texts = []
        for text in examples["text"]:
            if config.endoftext_handling == "remove":
                texts.append(text.replace("<|endoftext|>", ""))
            elif config.endoftext_handling == "replace_eos":
                texts.append(text.replace("<|endoftext|>", tokenizer.eos_token))
            elif config.endoftext_handling == "split":
                # Split text into multiple smaller texts
                texts.extend([t.strip() for t in text.split("<|endoftext|>") if t.strip()])
            else: # "keep"
                texts.append(text)
        return {"text": texts}

    # Apply preprocessing first if needed
    if config.endoftext_handling != "keep":
        dataset = dataset.map(handle_endoftext, batched=True, num_proc=os.cpu_count())

    # --- Tokenization and Chunking for inv-first format ---
    stride = int(config.max_seq_length * config.overlap)

    def tokenize_function_inv_first(examples):
        # First tokenize normally to get the tokens
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_length=True,
            add_special_tokens=False,  # Don't add BOS/EOS automatically
        )
        
        # For inv-first format, we need to:
        # 1. Keep the original first token
        # 2. Store it separately
        # 3. The sequence will have the real first token, not BOS
        
        processed_input_ids = []
        original_first_tokens = []
        
        for input_ids in tokenized["input_ids"]:
            if len(input_ids) > 0:
                # Store the original first token
                original_first_token = input_ids[0]
                original_first_tokens.append(original_first_token)
                
                # Keep the sequence as-is (with real first token)
                processed_input_ids.append(input_ids)
            else:
                # Handle empty sequences (shouldn't happen often)
                original_first_tokens.append(tokenizer.unk_token_id)
                processed_input_ids.append([tokenizer.unk_token_id])
        
        return {
            "input_ids": processed_input_ids,
            "original_first_token": original_first_tokens,
            "length": tokenized["length"]
        }

    # Note: `return_overflowing_tokens` creates a mapping from the original sample
    # to the new chunked samples. We must remove original columns to get a clean dataset.
    tokenized_dataset = dataset.map(
        tokenize_function_inv_first,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
    )
    
    # Filter out any chunks that are shorter than the max sequence length
    final_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) == config.max_seq_length)
    
    print(f"Chunking complete. Total chunks created: {len(final_dataset)}")
    print(f"Each chunk contains 'input_ids' and 'original_first_token' fields")
    return final_dataset


def upload_to_hub_inv_first(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    repo_id: str,
):
    """Saves the dataset and tokenizer and uploads them to the Hugging Face Hub."""
    print(f"Step 3: Uploading inv-first dataset to Hugging Face Hub at {repo_id}...")
    
    # Create repos on the Hub
    tokenizer_repo_id = f"{repo_id}-bpe-tokenizer"
    create_repo(repo_id, exist_ok=True, repo_type="dataset")
    create_repo(tokenizer_repo_id, exist_ok=True)
    
    # Upload tokenizer
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer.save_pretrained(temp_dir)
        upload_folder(
            repo_id=tokenizer_repo_id,
            folder_path=temp_dir,
            commit_message="Upload tokenizer for inv-first dataset",
        )
    print(f"Tokenizer uploaded to: {tokenizer_repo_id}")
    
    # Push dataset to the Hub
    dataset_dict.push_to_hub(repo_id, commit_message="Upload inv-first tokenized dataset")
    print(f"Dataset uploaded to: {repo_id}")


# --- Main Orchestrator ---

def create_inv_first_dataset(config: DataConfig):
    """Main function to orchestrate the inv-first dataset creation and upload process."""
    if not config.hf_user:
        raise ValueError("HF_USERNAME environment variable not set.")
        
    login_to_hf(config.hf_token)

    # --- Dataset Loading and Repo Naming ---
    dataset_id = config.hf_path.split("/")[-1]
    repo_name = (
        f"{config.hf_user}/{dataset_id}"
        f"-inv-first"
        f"-voc{config.vocabulary_size}"
        f"-seq{config.max_seq_length}"
        f"-overlap{int(config.overlap * 100)}"
    )
    
    # --- Main Pipeline ---
    # 1. Load initial dataset
    original_dataset = load_dataset(config.hf_path, config.config_name)

    # 2. Train tokenizer on the full text
    tokenizer = train_new_tokenizer(original_dataset, config)
    
    # 3. Combine all data, then tokenize and chunk together for inv-first format
    combined_dataset = concatenate_datasets([original_dataset[s] for s in original_dataset.keys()])
    processed_chunks = preprocess_and_chunk_dataset_inv_first(combined_dataset, tokenizer, config)
    
    # 4. Create final train/test/eval splits from the shuffled chunks
    processed_chunks = processed_chunks.shuffle(seed=42)
    
    if config.custom_splits:
        total_chunks = len(processed_chunks)
        train_size = int(total_chunks * config.train_split)
        test_size = int(total_chunks * config.test_split)
        
        final_dataset_splits = DatasetDict({
            "train": processed_chunks.select(range(train_size)),
            "test": processed_chunks.select(range(train_size, train_size + test_size)),
            "validation": processed_chunks.select(range(train_size + test_size, total_chunks)),
        })
    else:
        # If not creating custom splits, we can't guarantee the original split
        # proportions after chunking. We'll save the whole thing as "train".
        print("Warning: Custom splits are disabled. Saving all chunks to the 'train' split.")
        final_dataset_splits = DatasetDict({"train": processed_chunks})

    print(f"Final dataset splits: {final_dataset_splits}")
    
    # Print sample to verify format
    sample = final_dataset_splits["train"][0]
    print(f"\nSample from inv-first dataset:")
    print(f"Input IDs length: {len(sample['input_ids'])}")
    print(f"Original first token: {sample['original_first_token']}")
    print(f"First few tokens: {sample['input_ids'][:10]}")
    print(f"Decoded first few tokens: {tokenizer.decode(sample['input_ids'][:10])}")
    print(f"Original first token decoded: {tokenizer.decode([sample['original_first_token']])}")
    
    # 5. Upload artifacts to the Hub
    upload_to_hub_inv_first(final_dataset_splits, tokenizer, repo_name)
    
    print("\n--- Process Complete ---")
    print(f"Dataset Repo: https://huggingface.co/datasets/{repo_name}")
    print(f"Tokenizer Repo: https://huggingface.co/{repo_name}-bpe-tokenizer")
    print("\nDataset Format:")
    print("- 'input_ids': The tokenized sequence with the original first token")
    print("- 'original_first_token': The original first token that can be replaced with BOS at runtime")


if __name__ == "__main__":
    # Configure and run the pipeline
    pipeline_config = DataConfig(
        hf_path="fhswf/TinyStoriesV2_cleaned",
        vocabulary_size=2048,
        max_seq_length=256,
        overlap=0.25,
        endoftext_handling="keep",
        custom_splits=True,
    )
    
    create_inv_first_dataset(pipeline_config)
