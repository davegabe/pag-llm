import collections.abc
import logging
import pathlib
import shutil
import tempfile
import urllib.parse
from typing import Generator

import requests
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

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
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        text_column: str = 'text'
    ):
        """
        Dataset class for text data.

        Args:
            dataset (Dataset): The dataset to process.
            tokenizer (PreTrainedTokenizerFast): The tokenizer to use for encoding text.
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
        max_length: int = 512,
        input_ids_column: str = 'input_ids'
    ):
        """
        Dataset class for pre-tokenized data.

        Args:
            dataset (Dataset): The dataset containing pre-tokenized input_ids.
            max_length (int): The maximum length of the input text.
            input_ids_column (str): The column name of the input_ids data.
        """
        self.dataset = dataset
        self.max_length = max_length
        self.input_ids_column = input_ids_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> BatchType:
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
        input_ids_list = items[self.input_ids_column]
        
        # Check if attention_mask is available in the dataset
        has_attention_mask = 'attention_mask' in items
        attention_mask_list = items.get('attention_mask', None) if has_attention_mask else None
        
        # Convert to tensor and handle padding/truncation
        batch_input_ids = []
        batch_attention_masks = []
        
        for i, input_ids in enumerate(input_ids_list):
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            
            # Get attention mask from dataset if available
            if has_attention_mask and attention_mask_list is not None:
                attention_mask = attention_mask_list[i]
                if isinstance(attention_mask, list):
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            else:
                # Fallback: create attention mask (1 for real tokens, 0 for padding)
                attention_mask = torch.ones(len(input_ids), dtype=torch.long)
            
            # Truncate if necessary (both input_ids and attention_mask)
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            # Pad if necessary
            if len(input_ids) < self.max_length:
                padding_length = self.max_length - len(input_ids)
                # Pad on the left
                input_ids = torch.cat([torch.zeros(padding_length, dtype=torch.long), input_ids])
                attention_mask = torch.cat([torch.zeros(padding_length, dtype=torch.long), attention_mask])
            
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        input_ids = torch.stack(batch_input_ids)
        attention_mask = torch.stack(batch_attention_masks)
        
        # For causal LM, labels are the same as input_ids, shifted by 1
        labels = input_ids.clone()
        shift_labels = input_ids.roll(-1, dims=1)
        shift_labels[:, -1] = 0

        return BatchType(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels,
                         shift_labels=shift_labels)


def load_and_process_dataset(
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        overfit: bool = False,
        text_column: str = 'text'
) -> tuple[TextDataset | PreTokenizedDataset, TextDataset | PreTokenizedDataset]:
    """
    Load and process dataset for training.

    Args:
        dataset_config (DatasetConfig): The dataset configuration
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use for encoding text.
        max_length (int): The maximum length of the input text.
        overfit (bool): If True, use a small subset of the dataset for debugging.
        text_column (str): The column name of the text data.

    Returns:
        TextDataset | PreTokenizedDataset: The training dataset.
        TextDataset | PreTokenizedDataset: The evaluation dataset.
    """
    # Load the dataset
    raw_dataset = load_dataset(
        path=dataset_config.name,
        name=dataset_config.config,
        data_files=dataset_config.data_files,
    )

    # Check if the dataset is pre-tokenized (has 'input_ids' column)
    sample_data = raw_dataset[dataset_config.train_split][0]
    is_pretokenized = 'input_ids' in sample_data and text_column not in sample_data
    
    if is_pretokenized:
        has_attention_mask = 'attention_mask' in sample_data
        print(f"Detected pre-tokenized dataset with attention_mask: {has_attention_mask}")
    else:
        print("Detected text dataset, will tokenize on-the-fly")

    # Check if overfitting is enabled (for debugging purposes)
    if overfit:
        # Use only the first 32 samples for both train and eval
        split_overfit = Subset(
            raw_dataset[dataset_config.train_split],
            range(32)
        )
        raw_dataset[dataset_config.train_split] = split_overfit
        raw_dataset[dataset_config.eval_split] = split_overfit
        print(f"WARNING: Overfitting mode enabled. Using only {len(split_overfit)} samples for both train and eval.")
    
    # Check if the dataset has the specified splits
    if dataset_config.eval_split not in raw_dataset:
        # Create custom splits
        full_samples = raw_dataset[dataset_config.train_split]
        train_samples = int(len(full_samples) * 0.8)

        # Create Subset
        raw_dataset[dataset_config.train_split] = Subset(full_samples, range(train_samples))
        raw_dataset[dataset_config.eval_split] = Subset(full_samples, range(train_samples, len(full_samples)))

    # Create training dataset
    if is_pretokenized:
        train_dataset = PreTokenizedDataset(
            raw_dataset[dataset_config.train_split],
            max_length
        )
    else:
        train_dataset = TextDataset(
            raw_dataset[dataset_config.train_split],
            tokenizer,
            max_length,
            text_column
        )
    train_size = len(train_dataset)

    # Create evaluation dataset
    if is_pretokenized:
        eval_dataset = PreTokenizedDataset(
            raw_dataset[dataset_config.eval_split],
            max_length
        )
    else:
        eval_dataset = TextDataset(
            raw_dataset[dataset_config.eval_split],
            tokenizer,
            max_length,
            text_column
        )
    
    # Limit evaluation dataset to 10% of training dataset size
    target_eval_size = int(train_size * 0.1)
    if len(eval_dataset) > target_eval_size:
        print(f"WARNING: Evaluation dataset size is larger than 10% of training dataset size. "
              f"Limiting evaluation dataset to {target_eval_size} samples.")
        eval_dataset = Subset(eval_dataset, range(target_eval_size))

    # Log dataset sizes
    print(f'Train dataset size:\t{train_size}')
    print(f'Eval dataset size:\t{len(eval_dataset)}')

    return train_dataset, eval_dataset


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
