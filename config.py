"""
Configuration schemas for Hydra files in configs/
"""
from typing import Callable

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

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
    hidden_layer_index: int


@dataclass
class DatasetPrefixConfig:
    min_length: int
    max_length: int


@dataclass
class DatasetConfig:
    name: str
    train_split: str
    eval_split: str
    prefix: DatasetPrefixConfig
    num_workers: int
    config: str | None = None


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

# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)


def apply_config(config_name: str = 'base') -> Callable[[Callable[[Config], None]], None]:
    """
    Decorator FACTORY for main() to apply the Hydra configuration automatically.

    Args:
        config_name: (str) Name of the configuration file, inside the configs/ folder

    Returns:
        The actual decorator.
    """

    def decorator(main_func: Callable[[Config], None]) -> None:
        """
        Actual decorator function to apply the Hydra configuration.

        Args:
            main_func: main() function to run with the configuration

        Returns:
            The decorated main() that can be invoked with no arguments now.
        """
        def _main_wrapper(hydra_dict_config: DictConfig):
            # Instantiate all the classes
            config = instantiate(hydra_dict_config)

            # Check that there are no more DictConfig
            def _require_no_dict_config(object, depth=4):
                if depth == 0:
                    return
                assert not isinstance(object, DictConfig)
                for k in dir(config):
                    if k[0] != '_':
                        v = getattr(config, k)
                        _require_no_dict_config(v, depth=depth-1)
            _require_no_dict_config(config)

            main_func(config)

        return hydra.main(version_base=None, config_path='configs', config_name=config_name)(_main_wrapper)

    return decorator
