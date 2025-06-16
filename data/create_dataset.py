import os
import tempfile
from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from huggingface_hub import login
from huggingface_hub import create_repo, upload_folder, upload_file

user = os.getenv("HF_USERNAME", "DaveGabe")
# Login if not already
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    login()  # Will prompt for your token

def create_dataset(hf_path: str, config: str, vocabulary_size: int, max_seq_length: int, overlap: float = 0.25, include_text: bool = True, endoftext_handling: str = "keep", add_special_tokens: bool = True):
    """
    Create a dataset from a HF dataset with BPE tokenization and overlapping chunks.
    
    Args:
        hf_path (str): Name of the dataset to load (e.g., "wikitext").
        config (str): Configuration of the dataset (e.g., "wikitext-103-v1"). Can be None.
        vocabulary_size (int): Size of the vocabulary for BPE tokenizer.
        max_seq_length (int): Maximum sequence length for each chunk.
        overlap (float): Fraction of overlap between chunks, default is 0.25.
        include_text (bool): Whether to include original text alongside token IDs in the dataset.
        endoftext_handling (str): How to handle <|endoftext|> tokens. Options:
            - "keep": Keep as special token (recommended for most use cases)
            - "remove": Remove the token entirely
            - "replace_eos": Replace with </s> token
            - "split": Split stories at <|endoftext|> and process separately
        add_special_tokens (bool): Whether to use tokenizer-managed BOS/EOS tokens for each chunk.
            When True, uses tokenizer's built-in add_special_tokens=True for proper token handling.
            When False, uses raw token sliding windows without special tokens.
    """
    # Step 1: Load dataset
    dataset_id = hf_path.split("/")[-1]  # Get the dataset name from the full path
    dataset = load_dataset(hf_path, config)

    # Step 2: Train BPE tokenizer using temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()

        # Define special tokens based on endoftext handling
        special_tokens = [
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "<mask>",
        ]
        
        if endoftext_handling == "keep":
            special_tokens.append("<|endoftext|>")

        trainer = BpeTrainer(
            vocab_size=vocabulary_size,
            min_frequency=2,
            special_tokens=special_tokens,
        )

        # Save all text to a temporary file for tokenizer training
        text_file = os.path.join(temp_dir, f"{dataset_id}-complete-text.txt")
        print(f"Creating complete text file: {text_file}")
        print(f"Endoftext handling: {endoftext_handling}")
        
        # Build corpus file without special-case logic - do splitting only at chunking time
        with open(text_file, "w", encoding="utf-8") as f:
            for item in dataset["train"]:
                f.write(item["text"] + "\n")

        tokenizer.train([text_file], trainer)

        tokenizer_path = os.path.join(temp_dir, f"tokenizer-{vocabulary_size}-seq{max_seq_length}-overlap{int(overlap * 100)}.json")
        tokenizer.save(tokenizer_path)

        # Wrap in transformers-compatible tokenizer
        tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        special_tokens_dict = {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>"
        }
        
        if endoftext_handling == "keep":
            special_tokens_dict["additional_special_tokens"] = ["<|endoftext|>"]
        
        tok.add_special_tokens(special_tokens_dict)
        
        # Verify special tokens are properly set
        print(f"Special tokens map: {tok.special_tokens_map}")
        print(f"BOS token ID: {tok.bos_token_id}, EOS token ID: {tok.eos_token_id}")

        # Step 3: Chunk with overlap using tokenizer-managed special tokens
        block_size = max_seq_length
        stride = int(block_size * overlap)
        min_chunk_threshold = block_size // 4  # Only keep chunks with at least 25% of target length

        def chunk_overlap_batched(examples):
            """Batched version for better performance"""
            all_input_ids = []
            all_texts = []
            
            for text in examples["text"]:
                # Handle different endoftext processing
                if endoftext_handling == "split":
                    # Split stories at <|endoftext|> and process each story separately
                    stories = text.split("<|endoftext|>")
                    
                    for story in stories:
                        story = story.strip()
                        if not story:  # Skip empty stories
                            continue
                            
                        chunks, text_chunks = _process_text_to_chunks(
                            story, tok, block_size, stride, min_chunk_threshold, 
                            add_special_tokens, include_text
                        )
                        all_input_ids.extend(chunks)
                        if include_text:
                            all_texts.extend(text_chunks)
                else:
                    # Process text based on the selected handling method
                    processed_text = text
                    if endoftext_handling == "remove":
                        processed_text = text.replace("<|endoftext|>", "")
                    elif endoftext_handling == "replace_eos":
                        processed_text = text.replace("<|endoftext|>", "</s>")
                    # For "keep", text remains unchanged
                    
                    chunks, text_chunks = _process_text_to_chunks(
                        processed_text, tok, block_size, stride, min_chunk_threshold, 
                        add_special_tokens, include_text
                    )
                    all_input_ids.extend(chunks)
                    if include_text:
                        all_texts.extend(text_chunks)
            
            result = {"input_ids": all_input_ids}
            if include_text:
                result["text"] = all_texts
            return result

        def _process_text_to_chunks(text, tokenizer, block_size, stride, min_threshold, add_special, include_text):
            """Helper function to process text into tokenized chunks using tokenizer-managed special tokens"""
            chunks = []
            text_chunks = []
            
            if add_special:
                # Use tokenizer-managed special tokens with overlapping windows
                # First, get raw tokens without special tokens to calculate overlaps properly
                raw_tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                
                # Calculate step size for overlap
                # Since tokenizer will add BOS/EOS, we need to account for that in our window size
                effective_content_length = block_size - 2  # Reserve space for BOS/EOS
                step = max(1, effective_content_length - stride)
                
                for i in range(0, len(raw_tokens), step):
                    # Extract chunk of raw tokens
                    chunk_tokens = raw_tokens[i:i + effective_content_length]
                    
                    # Only keep chunks that meet minimum threshold
                    if len(chunk_tokens) >= min_threshold:
                        # Convert back to text for tokenizer processing
                        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                        
                        # Use tokenizer to add special tokens and pad to exact length
                        tokenized = tokenizer(
                            chunk_text,
                            add_special_tokens=True,
                            max_length=block_size,
                            truncation=True,
                            padding="max_length",
                            return_attention_mask=False
                        )
                        
                        chunks.append(tokenized["input_ids"])
                        if include_text:
                            # Decode with special tokens visible
                            text_chunk = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)
                            text_chunks.append(text_chunk)
            else:
                # Process without special tokens - use raw token sliding window
                raw_tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                
                for i in range(0, len(raw_tokens) - block_size + 1, block_size - stride):
                    chunk_tokens = raw_tokens[i:i + block_size]
                    if len(chunk_tokens) == block_size:
                        chunks.append(chunk_tokens)
                        if include_text:
                            text_chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                            text_chunks.append(text_chunk)
            
            return chunks, text_chunks

        # Process dataset with batched mapping for better performance using multiprocessing
        print("Tokenizing dataset with overlapping chunks...")
        tokenized = {}
        for split in dataset:
            tokenized_split = dataset[split].map(
                chunk_overlap_batched, 
                batched=True, 
                batch_size=100,  # Process in batches
                num_proc=os.cpu_count(),  # Use all available CPU cores
                remove_columns=["text"]
            )
            
            # No need to flatten since we return flat lists from the batched function
            tokenized[split] = tokenized_split

        tokenized_dataset = DatasetDict(tokenized)
        print(f"Tokenized dataset: {tokenized_dataset}")

        # Step 4: Save and push to Hugging Face Hub
        repo_name = user + f"/{dataset_id}-bpe-{vocabulary_size}-seq{max_seq_length}-overlap{int(overlap * 100)}"
        
        # Save tokenizer to temporary directory
        tokenizer_dir = os.path.join(temp_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tok.save_pretrained(tokenizer_dir)

        # Create repositories
        create_repo(repo_name + "-tokenizer", exist_ok=True)
        create_repo(repo_name, exist_ok=True)
        
        # Upload tokenizer
        upload_folder(repo_id=repo_name + "-tokenizer", folder_path=tokenizer_dir)
        
        # Upload complete text file to the main dataset repository  
        complete_text_filename = f"{dataset_id}-complete-text.txt"
        print(f"Uploading complete text file to {repo_name}")
        upload_file(
            path_or_fileobj=text_file,
            path_in_repo=complete_text_filename,
            repo_id=repo_name,
            repo_type="dataset"
        )

        # Push tokenized dataset
        print(f"Pushing tokenized dataset to {repo_name}")
        tokenized_dataset.push_to_hub(repo_name)
        
        print(f"Dataset creation complete!")
        print(f"Tokenized dataset: {repo_name}")
        print(f"Tokenizer: {repo_name}-tokenizer")
        print(f"Complete text file uploaded as: {complete_text_filename}")

if __name__ == "__main__":
    create_dataset(
        hf_path="fhswf/TinyStoriesV2_cleaned",
        config=None,
        vocabulary_size=2048,
        max_seq_length=256,
        overlap=0.25,
        include_text=True,
        endoftext_handling="keep",
        add_special_tokens=True  # Add BOS/EOS tokens to each chunk
    )