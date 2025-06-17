import os
import tempfile
from datasets import load_dataset, DatasetDict, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
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

def create_dataset(hf_path: str, config: str, vocabulary_size: int, max_seq_length: int, overlap: float = 0.25, include_text: bool = True, endoftext_handling: str = "keep", add_special_tokens: bool = True, train_split: float = 0.8, test_split: float = 0.1, eval_split: float = 0.1, custom_splits: bool = False):
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
        train_split (float): Fraction of data to use for training, default is 0.8.
        test_split (float): Fraction of data to use for testing, default is 0.1.
        eval_split (float): Fraction of data to use for evaluation, default is 0.1.
        custom_splits (bool): Whether to create custom train/test/eval splits from the data.
            When True, creates new splits based on the split ratios.
            When False, preserves the original dataset splits.
    """
    # Validate split ratios
    if custom_splits:
        if abs(train_split + test_split + eval_split - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_split + test_split + eval_split}")
        if any(split <= 0 for split in [train_split, test_split, eval_split]):
            raise ValueError("All split ratios must be positive")
    
    # Step 1: Load dataset
    dataset_id = hf_path.split("/")[-1]  # Get the dataset name from the full path
    dataset_name = f"/{dataset_id}-voc{vocabulary_size}-seq{max_seq_length}-overlap{int(overlap * 100)}"
    repo_name = user + dataset_name
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

        # Step 2.5: Create custom splits if requested
        if custom_splits:
            # Check if dataset already has validation/eval split
            has_validation = "validation" in dataset or "eval" in dataset or "val" in dataset
            
            if has_validation:
                print("Dataset already contains validation/eval split. Skipping custom splitting.")
                print(f"Using original dataset splits: {dataset}")
            else:
                print(f"Creating custom splits: train={train_split}, test={test_split}, eval={eval_split}")
                
                # Combine train and test splits as source data
                available_splits = list(dataset.keys())
                print(f"Available splits in dataset: {available_splits}")
                
                # Collect all available data for splitting
                all_data = [dataset[split] for split in available_splits]
                
                # Concatenate all source data
                if len(all_data) > 1:
                    source_data = concatenate_datasets(all_data)
                    print(f"Concatenated {len(all_data)} splits into source data")
                else:
                    source_data = all_data[0]
                
                total_examples = len(source_data)
                print(f"Total examples for splitting: {total_examples}")
                
                # Calculate split sizes
                train_size = int(total_examples * train_split)
                test_size = int(total_examples * test_split)
                eval_size = total_examples - train_size - test_size  # Remaining goes to eval
                
                print(f"Split sizes: train={train_size}, test={test_size}, eval={eval_size}")
                
                # Create the splits
                train_data = source_data.select(range(0, train_size))
                test_data = source_data.select(range(train_size, train_size + test_size))
                eval_data = source_data.select(range(train_size + test_size, total_examples))
                
                # Update the dataset with new splits
                dataset = DatasetDict({
                    "train": train_data,
                    "test": test_data,
                    "validation": eval_data  # Using "validation" as the standard name
                })
                
                print(f"Custom splits created: {dataset}")
        else:
            print(f"Using original dataset splits: {dataset}")

        tokenizer.train([text_file], trainer)

        # Add special tokens to tokenizer
        tokenizer.post_process = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ]
        )

        # Wrap in transformers-compatible tokenizer
        tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
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
                # Manually add BOS/EOS and handle padding

                # Get raw tokens from the input text without any special tokens added by the tokenizer yet
                raw_tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                
                # target_content_length is the number of tokens from raw_tokens that we want to place
                # between the BOS and EOS tokens.
                # This must be (block_size - 2) if BOS and EOS are present.
                target_content_length = block_size - 2
                
                # min_content_len is the minimum number of actual content tokens (excluding BOS/EOS)
                # required to form a valid chunk. This is based on the min_threshold passed to the function,
                # which in the original code, compared against the length of the content part.
                min_content_len = min_threshold # min_threshold is min_chunk_threshold

                # stride is the overlap of the full block_size, as in the original logic.
                # step determines how much the window for extracting content_tokens slides over raw_tokens.
                # It's based on the target_content_length and the desired stride.
                step = max(1, target_content_length - stride if target_content_length > stride else 1)

                if target_content_length < 0:
                    # This implies block_size < 2, which is too small to add both BOS and EOS.
                    # In this scenario, we'll effectively skip adding special tokens for this text segment,
                    # or one might consider falling back to the add_special=False logic.
                    # For now, we'll log a warning and produce no chunks for this text if this occurs.
                    # print(f"Warning: block_size {block_size} is too small to add BOS and EOS. Skipping for this text segment.")
                    pass # No chunks will be generated from this 'text' if target_content_length < 0

                else:
                    for i in range(0, len(raw_tokens), step):
                        # Extract the segment of raw_tokens that will form the content of our chunk
                        current_content_ids = raw_tokens[i : i + target_content_length]
                        
                        # Check if the extracted content meets the minimum length requirement
                        if len(current_content_ids) >= min_content_len:
                            # Manually construct the chunk with BOS and EOS tokens
                            final_chunk_ids = []
                            if tokenizer.bos_token_id is not None:
                                final_chunk_ids.append(tokenizer.bos_token_id)
                            
                            final_chunk_ids.extend(current_content_ids)
                            
                            if tokenizer.eos_token_id is not None:
                                final_chunk_ids.append(tokenizer.eos_token_id)
                            
                            # Pad the chunk to the required block_size
                            padding_needed = block_size - len(final_chunk_ids)
                            
                            if padding_needed >= 0:
                                padded_chunk = final_chunk_ids + [tokenizer.pad_token_id] * padding_needed
                            else:
                                # This case (padding_needed < 0) should ideally not be hit if
                                # len(current_content_ids) <= target_content_length and target_content_length = block_size - 2.
                                # It implies len(final_chunk_ids) > block_size.
                                # For robustness, truncate.
                                padded_chunk = final_chunk_ids[:block_size]
                                # Attempt to preserve BOS/EOS if they were overwritten by truncation
                                if tokenizer.bos_token_id is not None and len(padded_chunk) > 0 and padded_chunk[0] != tokenizer.bos_token_id:
                                    padded_chunk[0] = tokenizer.bos_token_id
                                if tokenizer.eos_token_id is not None and len(padded_chunk) == block_size and padded_chunk[-1] != tokenizer.eos_token_id:
                                    if block_size > 1: # Ensure there's space for EOS if BOS is also present
                                        padded_chunk[-1] = tokenizer.eos_token_id
                                    elif block_size == 1 and tokenizer.bos_token_id is None : # Only EOS if block_size is 1 and no BOS
                                        padded_chunk[0] = tokenizer.eos_token_id


                            # Ensure the final chunk is exactly block_size, as a final safety measure
                            if len(padded_chunk) != block_size:
                                if len(padded_chunk) > block_size:
                                    padded_chunk = padded_chunk[:block_size]
                                else: # len(padded_chunk) < block_size
                                    padded_chunk.extend([tokenizer.pad_token_id] * (block_size - len(padded_chunk)))
                            
                            chunks.append(padded_chunk)

                            if include_text:
                                # Decode the manually constructed and padded chunk
                                text_chunk = tokenizer.decode(padded_chunk, skip_special_tokens=False)
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
        tokenized_dataset.save_to_disk(f".{dataset_name}")
        print(f"Dataset saved to {dataset_name}")
        
        # Save tokenizer to temporary directory
        tokenizer_dir = os.path.join(temp_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tok.save_pretrained(tokenizer_dir)

        # Create repositories
        create_repo(repo_name + "-bpe-tokenizer", exist_ok=True)
        
        # Upload tokenizer
        upload_folder(repo_id=repo_name + "-bpe-tokenizer", folder_path=tokenizer_dir)
        
        # Upload complete text file to the main dataset repository  
        if include_text:
            complete_text_filename = f"{dataset_id}-complete-text.txt"
            print(f"Uploading complete text file to {repo_name}")
            create_repo(repo_name, exist_ok=True)
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
        print(f"Tokenizer: {repo_name}-bpe-tokenizer")

if __name__ == "__main__":
    create_dataset(
        hf_path="fhswf/TinyStoriesV2_cleaned",
        config=None,
        vocabulary_size=2048,
        max_seq_length=256,
        overlap=0.25,
        include_text=False,
        endoftext_handling="keep",
        add_special_tokens=True,  # Add BOS/EOS tokens to each chunk
        train_split=0.8,
        test_split=0.1,
        eval_split=0.1,
        custom_splits=True  # Enable custom train/test/eval splits
    )