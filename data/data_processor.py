import logging
import pathlib
import shutil
import tempfile
from typing import TypedDict
import urllib.parse

import requests
from datasets import load_dataset, IterableDataset
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from config import DatasetConfig, TrainingConfig

logger = logging.getLogger(__name__)

class BatchType(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

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
        batched_item = self.__getitems__([idx])
        return batched_item[0]

    def __getitems__(self, indices: list[int]) -> list[BatchType]:
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

        # For causal LM, labels are the same as input_ids, shifted by 1
        encodings['labels'] = encodings['input_ids'].roll(-1, dims=1)
        encodings['labels'][:, -1] = 0

        # Make it a list of single samples, as required by default DataLoader collate_fn
        return [
            {
                k: v[i]
                for k, v in encodings.items()
            }
            for i in range(len(texts))
        ]


def load_and_process_dataset(
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        text_column: str = 'text'
) -> tuple[TextDataset, TextDataset]:
    """
    Load and process dataset for training.

    Args:
        dataset_config (DatasetConfig): The dataset configuration
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use for encoding text.
        max_length (int): The maximum length of the input text.
        text_column (str): The column name of the text data.

    Returns:
        TextDataset: The training dataset.
        TextDataset: The evaluation dataset.
    """
    # Load the dataset
    raw_dataset = load_dataset(
        path=dataset_config.name,
        name=dataset_config.config,
        data_files=dataset_config.data_files,
    )

    # Create training dataset
    train_dataset = TextDataset(
        raw_dataset[dataset_config.train_split],
        tokenizer,
        max_length,
        text_column
    )
    train_size = len(train_dataset)

    # Create evaluation dataset
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


def load_and_process_dataloader(
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        text_column: str = 'text',
) -> tuple[DataLoader, DataLoader]:
    train_dataset, eval_dataset = load_and_process_dataset(dataset_config, tokenizer, max_length, text_column)

    # Create the dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.batch_size,
        shuffle=not isinstance(train_dataset, IterableDataset),  # You cannot shuffle an IterableDataset (streaming)
        num_workers=dataset_config.num_workers,
        collate_fn=lambda x: x,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=dataset_config.num_workers,
        collate_fn=lambda x: x,
    )

    return train_dataloader, eval_dataloader


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
