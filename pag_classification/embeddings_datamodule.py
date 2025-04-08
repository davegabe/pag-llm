import pathlib

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from config import SentenceClassificationConfig
from pag_classification.embeddings_dataset import SentenceEmbeddingsDataset

SENTENCE_TO_EMBEDDING_DIR = 'sentence-to-embedding'
SENTENCE_TO_EMBEDDING_TRAIN_FILENAME = 'train_embeddings.pt'
SENTENCE_TO_EMBEDDING_TEST_FILENAME = 'test_embeddings.pt'


class SentenceEmbeddingsDataModule(LightningDataModule):
    train_dataset: SentenceEmbeddingsDataset
    test_dataset: SentenceEmbeddingsDataset
    val_dataset: SentenceEmbeddingsDataset

    def __init__(self, config: SentenceClassificationConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size

    @staticmethod
    def get_train_split_embeddings_file(config: SentenceClassificationConfig) -> pathlib.Path:
        return config.output_dir / SENTENCE_TO_EMBEDDING_DIR / SENTENCE_TO_EMBEDDING_TRAIN_FILENAME

    @staticmethod
    def get_test_split_embeddings_file(config: SentenceClassificationConfig) -> pathlib.Path:
        return config.output_dir / SENTENCE_TO_EMBEDDING_DIR / SENTENCE_TO_EMBEDDING_TEST_FILENAME

    def setup(self, stage: str = None) -> None:
        self.train_dataset = SentenceEmbeddingsDataset(
            SentenceEmbeddingsDataModule.get_train_split_embeddings_file(self.config)
        )
        self.test_dataset = SentenceEmbeddingsDataset(
            SentenceEmbeddingsDataModule.get_test_split_embeddings_file(self.config)
        )
        self.val_dataset = self.test_dataset #! TODO: use a different dataset for validation

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
