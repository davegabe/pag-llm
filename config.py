"""
Configuration schemas for Hydra files in config/
"""

from omegaconf import OmegaConf

import pathlib
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    max_seq_length: int


@dataclass
class ModelConfig:
    pretrained_base: str
    output_dir: pathlib.Path


@dataclass
class DatasetConfig:
    name: str
    config: str | None
    train_split: str
    eval_split: str


@dataclass
class LoggingConfig:
    logging_steps: int
    evaluation_steps: int
    save_steps: int


@dataclass
class Config:
    training: TrainingConfig
    model: ModelConfig
    dataset: DatasetConfig
    logging: LoggingConfig


# Register a custom resolver for paths
OmegaConf.register_new_resolver("path", pathlib.Path)
