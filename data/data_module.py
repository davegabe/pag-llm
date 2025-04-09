import pathlib

import lightning as pl
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from config import LLMPagConfig
from data.data_processor import load_and_process_dataset, download_files


class LMDataModule(pl.LightningDataModule):
    def __init__(self, config: LLMPagConfig, tokenizer: PreTrainedTokenizerFast):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.val_dataset = None
    
    def prepare_data(self):
        """Download data files if needed. This method is called only once on rank 0."""
        if self.config.dataset.files_to_download:
            # Download files if needed
            download_files(
                self.config.dataset.files_to_download,
                destination_dir=pathlib.Path('./downloaded_dataset')
            )
    
    def setup(self, stage=None):
        """Load datasets"""
        self.train_dataset, self.val_dataset = load_and_process_dataset(
            self.config.dataset,
            self.tokenizer,
            self.config.training.max_seq_length
        )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
            collate_fn=lambda x: x,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.dataset.num_workers,
            collate_fn=lambda x: x,
        )
