import os
from datasets import load_dataset, Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from huggingface_hub import login
from huggingface_hub import create_repo, upload_folder
from datasets import Dataset, DatasetDict

user = os.getenv("HF_USERNAME", "DaveGabe")
# Login if not already
login()  # Will prompt for your token

def create_dataset(name: str, config: str, vocabulary_size: int, max_seq_length: int, overlap: float = 0.25):
    """
    Create a dataset from the Wikitext-103 dataset with BPE tokenization and overlapping chunks.
    
    Args:
        name (str): Name of the dataset to load (e.g., "wikitext").
        config (str): Configuration of the dataset (e.g., "wikitext-103-v1").
        vocabulary_size (int): Size of the vocabulary for BPE tokenizer.
        max_seq_length (int): Maximum sequence length for each chunk.
        overlap (float): Fraction of overlap between chunks, default is 0.25.
    """
    # Step 1: Load dataset
    dataset_name = name.split("/")[-1]  # Get the dataset name from the full path
    dataset = load_dataset(name, config)

    # Step 2: Train BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=2,
        special_tokens=[
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "<mask>",
        ],
    )

    # Save all text to a temporary file to train tokenizer
    text_file =  "temp_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        for item in dataset["train"]:
            f.write(item["text"] + "\n")

    tokenizer.train([text_file], trainer)

    tokenizer_path = f"tokenizer-{vocabulary_size}-seq{max_seq_length}-overlap{int(overlap * 100)}.json"
    tokenizer.save(tokenizer_path)

    # Wrap in transformers-compatible tokenizer
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tok.add_special_tokens({
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "mask_token": "<mask>"
    })

    # Step 3: Chunk with overlap
    block_size = max_seq_length
    stride = int(block_size * overlap)

    def chunk_overlap(example):
        tokens = tok(example["text"], return_attention_mask=False)["input_ids"]
        chunks = []
        for i in range(0, len(tokens) - block_size + 1, block_size - stride):
            chunk = tokens[i:i + block_size]
            if len(chunk) == block_size:
                chunks.append(chunk)
        return {"input_ids": chunks}

    # Flatten map to get individual chunks
    print("Tokenizing dataset with overlapping chunks...")
    tokenized = {}
    for split in dataset:
        tokenized_split = dataset[split].map(chunk_overlap, batched=False, remove_columns=["text"])
        # Flatten the nested list of chunks
        all_chunks = sum(tokenized_split["input_ids"], [])
        tokenized[split] = Dataset.from_dict({"input_ids": all_chunks})

    tokenized_dataset = DatasetDict(tokenized)
    print(f"Tokenized dataset: {tokenized_dataset}")

    # Step 4: Save and push to Hugging Face Hub
    repo_name = user + f"/{dataset_name}-bpe-{vocabulary_size}-seq{max_seq_length}-overlap{int(overlap * 100)}"
    tokenizer_dir = "./tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tok.save_pretrained(tokenizer_dir)

    create_repo(repo_name + "-tokenizer", exist_ok=True)
    upload_folder(repo_id=repo_name + "-tokenizer", folder_path=tokenizer_dir)

    # Push dataset
    tokenized_dataset.push_to_hub(repo_name)

if __name__ == "__main__":
    # Example usage
    create_dataset(
        name="fhswf/TinyStoriesV2_cleaned",
        config=None,
        vocabulary_size=2048,
        max_seq_length=512,
        overlap=0.25
    )