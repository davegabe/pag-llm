import os
import tempfile
from dataclasses import dataclass
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from transformers import PreTrainedTokenizerFast
from huggingface_hub import login, create_repo, upload_folder

# --- Configuration ---
@dataclass
class DataConfig:
    """Configuration for the dataset creation process."""
    hf_path: str = "fhswf/TinyStoriesV2_cleaned"
    config_name: str | None = None
    vocabulary_size: int = 2048
    max_seq_length: int = 256
    overlap: float = 0.25
    
    # "keep", "remove", "replace_eos", or "split"
    endoftext_handling: str = "keep"
    
    custom_splits: bool = True
    train_split: float = 0.8
    test_split: float = 0.1
    # 1.0 - (train_split + test_split) is validation split
    
    # Hugging Face Hub configuration
    hf_user: str | None = os.getenv("HF_USERNAME")
    hf_token: str | None = os.getenv("HF_TOKEN")

# --- Helper Functions ---

def login_to_hf(token: str | None):
    """Logs into the Hugging Face Hub."""
    if token:
        login(token=token)
    else:
        print("HF_TOKEN not found. Attempting interactive login.")
        login()

def train_new_tokenizer(dataset: DatasetDict, config: DataConfig) -> PreTrainedTokenizerFast:
    """Trains a new BPE tokenizer from the dataset text."""
    print("Step 1: Training new tokenizer...")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as temp_f:
        # Combine all splits for robust tokenizer training
        full_text_dataset = concatenate_datasets([dataset[s] for s in dataset.keys()])
        for item in full_text_dataset["text"]:
            temp_f.write(item + "\n")
        temp_filename = temp_f.name

    # Setup tokenizer and trainer
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
    if config.endoftext_handling == "keep":
        special_tokens.append("<|endoftext|>")
        
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Metaspace()
    trainer = BpeTrainer(vocab_size=config.vocabulary_size, min_frequency=2, special_tokens=special_tokens)
    
    # Train
    tokenizer.train([temp_filename], trainer)
    os.remove(temp_filename)

    # Wrap in PreTrainedTokenizerFast for transformers compatibility
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=True,
    )
    if config.endoftext_handling == "keep":
        fast_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endoftext|>"]})
        
    print("Tokenizer training complete.")
    return fast_tokenizer


def preprocess_and_chunk_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerFast, config: DataConfig) -> Dataset:
    """Preprocesses text and chunks the dataset using the tokenizer's sliding window."""
    print("Step 2: Preprocessing and chunking dataset...")
    
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

    # --- Tokenization and Chunking ---
    stride = int(config.max_seq_length * config.overlap)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            stride=stride,
            return_overflowing_tokens=True, # This is the magic that creates the chunks
            return_length=True,
        )

    # Note: `return_overflowing_tokens` creates a mapping from the original sample
    # to the new chunked samples. We must remove original columns to get a clean dataset.
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
    )
    
    # Filter out any chunks that are shorter than the max sequence length
    # This often happens with the very last chunk of a document.
    final_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) == config.max_seq_length)
    
    print(f"Chunking complete. Total chunks created: {len(final_dataset)}")
    return final_dataset


def upload_to_hub(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    repo_id: str,
):
    """Saves the dataset and tokenizer and uploads them to the Hugging Face Hub."""
    print(f"Step 3: Uploading to Hugging Face Hub at {repo_id}...")
    
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
            commit_message="Upload tokenizer",
        )
    print(f"Tokenizer uploaded to: {tokenizer_repo_id}")
    
    # Push dataset to the Hub
    dataset_dict.push_to_hub(repo_id, commit_message="Upload tokenized dataset")
    print(f"Dataset uploaded to: {repo_id}")


# --- Main Orchestrator ---

def create_processed_dataset(config: DataConfig):
    """Main function to orchestrate the dataset creation and upload process."""
    if not config.hf_user:
        raise ValueError("HF_USERNAME environment variable not set.")
        
    login_to_hf(config.hf_token)

    # --- Dataset Loading and Repo Naming ---
    dataset_id = config.hf_path.split("/")[-1]
    repo_name = (
        f"{config.hf_user}/{dataset_id}"
        f"-voc{config.vocabulary_size}"
        f"-seq{config.max_seq_length}"
        f"-overlap{int(config.overlap * 100)}"
    )
    
    # --- Main Pipeline ---
    # 1. Load initial dataset
    original_dataset = load_dataset(config.hf_path, config.config_name)

    # 2. Train tokenizer on the full text
    tokenizer = train_new_tokenizer(original_dataset, config)
    
    # 3. Combine all data, then tokenize and chunk together
    # This ensures consistency and prevents data leakage across future splits.
    combined_dataset = concatenate_datasets([original_dataset[s] for s in original_dataset.keys()])
    processed_chunks = preprocess_and_chunk_dataset(combined_dataset, tokenizer, config)
    
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
    
    # 5. Upload artifacts to the Hub
    upload_to_hub(final_dataset_splits, tokenizer, repo_name)
    
    print("\n--- Process Complete ---")
    print(f"Dataset Repo: https://huggingface.co/datasets/{repo_name}")
    print(f"Tokenizer Repo: https://huggingface.co/{repo_name}-bpe-tokenizer")


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
    
    create_processed_dataset(pipeline_config)