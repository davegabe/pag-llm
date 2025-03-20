"""
Configuration schemas for Hydra files in configs/
"""
from typing import Callable

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig, ListConfig

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
    files_to_download: list[str] | None = None
    data_files: dict[str, list[str] | str] | None = None

    def __post_init__(self):
        if self.files_to_download is not None and isinstance(self.files_to_download, ListConfig):
            self.files_to_download = list(self.files_to_download)

        if self.data_files is not None and isinstance(self.data_files, DictConfig):
            self.data_files = {
                k: list(v) if isinstance(v, ListConfig) else v
                for k, v in self.data_files.items()
            }


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
            def _require_no_dict_config(obj, depth=4):
                """
                Check that no DictConfig is used in the final object.

                This often causes LOTS of very subtle problems.
                For instance, load_dataset takes data_files, which can be a dict.
                So it is pretty natural to add a dictionary to the YAML file.

                Then, you take the config object.
                But config.data_files is NOT a dict, being a DictConfig object instead.
                This causes load_dataset NOT to work anymore, with a non-interpretable error.
                Also, if you print config.data_files to double-check, it LOOKS LIKE a real dict, while it is not!

                Args:
                    obj: Object to be checked not to have DictConfig fields, recursively.
                    depth: Depth of the recursion.
                """
                assert not isinstance(obj, DictConfig), 'Found a residual of DictConfig in the Hydra config file'
                assert not isinstance(obj, ListConfig), 'Found a residual of ListConfig in the Hydra config file'

                if depth == 0:
                    return
                for k in dir(obj):
                    if k[0] != '_':
                        v = getattr(obj, k)
                        _require_no_dict_config(v, depth=depth-1)
            _require_no_dict_config(config)

            main_func(config)

        return hydra.main(version_base=None, config_path='configs', config_name=config_name)(_main_wrapper)

    return decorator
