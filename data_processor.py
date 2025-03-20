from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from .config import DatasetConfig


class TextDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizerFast, max_length: int = 512,
                 text_column: str = "text"):
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

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_column]  # Get text from the specified column

        # Tokenize text
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # For causal LM, labels are the same as input_ids
        encodings["labels"] = encodings["input_ids"].clone()

        # Convert to correct shape (squeeze out batch dimension)
        for key in encodings:
            encodings[key] = encodings[key].squeeze(0)

        return encodings


def load_and_process_dataset(
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        text_column: str = "text"
):
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
    if dataset_config.config:
        raw_dataset = load_dataset(
            dataset_config.name,
            dataset_config.config,
        )
    else:
        raw_dataset = load_dataset(dataset_config.name)

    train_dataset = TextDataset(
        raw_dataset[dataset_config.train_split],
        tokenizer,
        max_length,
        text_column
    )
    eval_dataset = TextDataset(
        raw_dataset[dataset_config.eval_split],
        tokenizer,
        max_length,
        text_column
    )

    return train_dataset, eval_dataset
