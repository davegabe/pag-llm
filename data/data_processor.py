import collections.abc
import logging
import pathlib
import re
import shutil
import tempfile
import urllib.parse
from typing import Generator

import requests
import torch
from datasets import load_dataset, load_from_disk
from torch import Tensor
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from config import DatasetConfig

logger = logging.getLogger(__name__)


class BatchType(collections.abc.Iterable):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    shift_labels: torch.Tensor

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                 shift_labels: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.shift_labels = shift_labels

    def to(self, device: torch.device) -> 'BatchType':
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.labels = self.labels.to(device)
        self.shift_labels = self.shift_labels.to(device)
        return self

    def __getitem__(self, batch_index: int) -> 'BatchType':
        return self.__class__(input_ids=self.input_ids[batch_index],
                              attention_mask=self.attention_mask[batch_index],
                              labels=self.labels[batch_index],
                              shift_labels=self.shift_labels[batch_index])

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'labels': self.labels,
            'shift_labels': self.shift_labels,
        }

    def __len__(self) -> int:
        if len(self.input_ids) == 1:
            return 1
        else:
            return len(self.input_ids)

    def __iter__(self) -> Generator[Tensor, None, None]:
        yield self.input_ids
        yield self.attention_mask
        yield self.labels
        yield self.shift_labels


class TextDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        text_column: str = 'text'
    ):
        """
        Dataset class for text data.

        Args:
            dataset (Dataset): The dataset to process.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding text.
            max_length (int): The maximum length of the input text.
            text_column (str): The column name of the text data.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

    def __len__(self):
        # noinspection PyTypeChecker
        return len(self.dataset)

    def __getitem__(self, idx: int) -> BatchType:
        #
        # All this mess is because when you do dataset[indexes], still __getitem__ is called
        is_batch = False
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, torch.Tensor) and idx.ndim == 0:
            idx = [idx.item()]
        else:
            is_batch = True

        # Otherwise, it is already a tensor / list of indices
        batch_type = self.__getitems__(idx)
        if is_batch:
            return batch_type
        else:
            return batch_type[0]

    def __iter__(self) -> Generator[BatchType, None, None]:
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __getitems__(self, indices: list[int] | torch.Tensor) -> BatchType:
        items = self.dataset[indices]
        texts = items[self.text_column]

        # Tokenize text
        encodings = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        # For causals LM, labels are the same as input_ids, shifted by 1
        # However:
        # - 'labels' should be the same as input_ids
        # - 'shift_labels' should be the actual shifted labels, output of the lm_head as the next token
        encodings['labels'] = encodings['input_ids'].clone()
        encodings['shift_labels'] = encodings['input_ids'].roll(-1, dims=1)
        encodings['shift_labels'][:, -1] = 0

        return BatchType(input_ids=encodings['input_ids'],
                         attention_mask=encodings['attention_mask'],
                         labels=encodings['labels'],
                         shift_labels=encodings['shift_labels'])


class PreTokenizedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        token_column: str = 'input_ids',
        attention_mask_column: str = 'attention_mask'
    ):
        """
        Dataset class for pre-tokenized data.

        Args:
            dataset (Dataset): The pre-tokenized dataset to process.
            tokenizer (PreTrainedTokenizerBase): The tokenizer for compatibility (mainly for pad_token_id).
            max_length (int): The maximum length of the input sequences.
            token_column (str): The column name of the tokenized data.
            attention_mask_column (str): The column name of the attention mask data.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_column = token_column
        self.attention_mask_column = attention_mask_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> BatchType:
        # Handle batch vs single item similar to TextDataset
        is_batch = False
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, torch.Tensor) and idx.ndim == 0:
            idx = [idx.item()]
        else:
            is_batch = True

        batch_type = self.__getitems__(idx)
        if is_batch:
            return batch_type
        else:
            return batch_type[0]

    def __iter__(self) -> Generator[BatchType, None, None]:
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __getitems__(self, indices: list[int] | torch.Tensor) -> BatchType:
        items = self.dataset[indices]
        token_ids = items[self.token_column]
        
        # Convert to tensor if needed and ensure proper shape
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # If single sequence, add batch dimension
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        
        # Get attention mask from dataset if available
        if self.attention_mask_column in items:
            attention_mask = items[self.attention_mask_column]
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        else:
            # Fallback: create attention mask manually
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            attention_mask = (token_ids != pad_token_id).long()
        
        # Ensure sequences are properly sized
        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_length:
            token_ids = token_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
        elif seq_len < self.max_length:
            pad_length = self.max_length - seq_len
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            # Pad token_ids
            padding = torch.full((batch_size, pad_length), pad_token_id, dtype=torch.long)
            token_ids = torch.cat([padding, token_ids], dim=1)  # Left padding
            
            # Pad attention_mask
            mask_padding = torch.zeros((batch_size, pad_length), dtype=torch.long)
            attention_mask = torch.cat([mask_padding, attention_mask], dim=1)  # Left padding
        
        # For causal LM, labels are the same as input_ids, shifted by 1
        labels = token_ids.clone()
        shift_labels = token_ids.roll(-1, dims=1)
        shift_labels[:, -1] = 0  # Set last token to 0 (or could use pad_token_id)

        return BatchType(
            input_ids=token_ids,
            attention_mask=attention_mask,
            labels=labels,
            shift_labels=shift_labels
        )


def load_and_process_dataset(
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        text_column: str = 'text'
) -> tuple[TextDataset | PreTokenizedDataset, TextDataset | PreTokenizedDataset, TextDataset | PreTokenizedDataset]:
    """
    Load and process dataset for training.

    Args:
        dataset_config (DatasetConfig): The dataset configuration
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding text.
        max_length (int): The maximum length of the input text.
        text_column (str): The column name of the text data.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            test_dataset: The test dataset.
    """
    # Check if we should use pre-tokenized dataset
    if dataset_config.use_pretokenized:
        # Determine if we should load from local path or remote
        if dataset_config.local_dataset_path:
            print(f"Loading pre-tokenized dataset from local path: {dataset_config.local_dataset_path}")
            raw_dataset = load_from_disk(dataset_config.local_dataset_path)
        elif dataset_config.pretokenized_dataset_name:
            print(f"Loading pre-tokenized dataset from remote: {dataset_config.pretokenized_dataset_name}")
            raw_dataset = load_dataset(dataset_config.pretokenized_dataset_name)
        else:
            raise ValueError("For pretokenized datasets, either 'local_dataset_path' or 'pretokenized_dataset_name' must be specified")

        # Create training dataset
        train_dataset = PreTokenizedDataset(
            raw_dataset[dataset_config.train_split],
            tokenizer,
            max_length,
        )
        train_size = len(train_dataset)
        
        # Create validation dataset
        val_dataset = PreTokenizedDataset(
            raw_dataset[dataset_config.eval_split],
            tokenizer,
            max_length,
        )
        
        # Create test dataset - use test_split if different from eval_split, otherwise reuse val_dataset
        if dataset_config.test_split != dataset_config.eval_split and dataset_config.test_split in raw_dataset:
            test_dataset = PreTokenizedDataset(
                raw_dataset[dataset_config.test_split],
                tokenizer,
                max_length,
            )
        else:
            raise ValueError(f"Test split '{dataset_config.test_split}' not found in pre-tokenized dataset")
            test_dataset = val_dataset
    else:
        # Use regular text dataset processing
        print(f"Loading text dataset: {dataset_config.name}")
        
        # Load the dataset
        raw_dataset = load_dataset(
            path=dataset_config.name,
            name=dataset_config.config,
            data_files=dataset_config.data_files,
        )

        # Check if the dataset has the specified splits
        if dataset_config.eval_split not in raw_dataset:
            # Create custom splits
            full_samples = raw_dataset[dataset_config.train_split]
            train_samples = int(len(full_samples) * 0.9) # 90% for training
            val_samples = int(len(full_samples) * 0.1) # 10% for validation + test

            # Create Subset
            raw_dataset[dataset_config.train_split] = Subset(full_samples, range(train_samples))
            raw_dataset[dataset_config.eval_split] = Subset(full_samples, range(train_samples, val_samples))

        # Create training dataset
        train_dataset = TextDataset(
            raw_dataset[dataset_config.train_split],
            tokenizer,
            max_length,
            text_column
        )
        train_size = len(train_dataset)

        # Create validation dataset
        val_dataset = TextDataset(
            raw_dataset[dataset_config.eval_split],
            tokenizer,
            max_length,
            text_column
        )
        
        # Create test dataset - use test_split if different from eval_split, otherwise reuse val_dataset
        if dataset_config.test_split != dataset_config.eval_split and dataset_config.test_split in raw_dataset:
            test_dataset = TextDataset(
                raw_dataset[dataset_config.test_split],
                tokenizer,
                max_length,
                text_column
            )
        else:
            test_dataset = val_dataset

    # Log dataset sizes
    print(f'Train dataset size:\t{train_size}')
    print(f'Validation dataset size:\t{len(val_dataset)}')
    print(f'Test dataset size:\t{len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset


def download_files(files_to_download: list[str], destination_dir: pathlib.Path, chunk_size=64*1024):
    destination_dir = destination_dir.resolve().expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)

    for file_url_string in files_to_download:
        filename = urllib.parse.urlparse(file_url_string).path.rsplit('/', maxsplit=1)[-1]
        destination_file = destination_dir / filename
        if destination_file.exists():
            continue

        # Download the file
        response = requests.get(file_url_string, stream=True, allow_redirects=True)
        if response.status_code != 200:
            raise RuntimeError(f'Error downloading file {file_url_string}: got status code {response.status_code}')

        # Prepare a progress bar
        file_size = int(response.headers.get('Content-Length', 0))

        # First, download the file into a temporary location,
        # so that we won't save broken data if interrupted while downloading
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with tqdm(desc=filename, total=file_size, unit='iB', unit_scale=True, unit_divisor=1024) as progress_bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = temp_file.write(data)
                progress_bar.update(size)

        # Now, move it to the final destination
        temp_file.flush()
        temp_file.close()
        shutil.move(temp_file.name, str(destination_file.absolute()))

        logger.info(f'{file_url_string} successfully downloaded.')


def clean_text(text: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """
    Clean the input text by removing special tokens and normalizing whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    # Remove all special tokens from attack texts
    for tok in tokenizer.all_special_tokens + ["<|endoftext|>"]:
        text = text.replace(tok, '')
    # Remove all the whitespaces since we have metaspace
    text = re.sub(r'\s+', '', text)
    # Replace metaspace with regular space
    text = text.replace("\u2581", " ").strip()
    return text