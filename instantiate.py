import pathlib

import torch

import models.loader as loader
from config import LLMPagConfig, CustomLLMPagConfig
from data.data_module import LMDataModule
from models.base_model import BaseLMModel
from models.identity_grad_embeddings_model import IdentityGradEmbeddingsModel
from models.identity_grad_model import IdentityGradModel
from models.inv_first_token import InvFirstTokenModel
from models.masked_embeddings_grad_model import MaskedIdentityGradEmbeddingsModel
from models.pag_hidden_model import PAGHiddenModel
from utils.index_token_to_dataset_item import DatasetIndexByToken


def instantiate_model_by_config(cfg: LLMPagConfig | CustomLLMPagConfig) -> tuple[BaseLMModel, LMDataModule, str]:
    """
    Instantiate the model and data module based on the provided configuration.

    Args:
        cfg: configuration object containing model and data settings.

    Returns:
        tuple: A tuple containing the instantiated model, data module, and model name.
    """
    # Load tokenizer and model
    if isinstance(cfg, CustomLLMPagConfig):
        model, tokenizer = loader.create_model_and_tokenizer(
            cfg.dataset,
            cfg.model
        )
        model_name = str(cfg.model.output_dir).split("/")[-1]
    else:
        model, tokenizer = loader.load_model_and_tokenizer(
            cfg.model.pretrained_base,
            cfg.model.random_initialization,
            cfg.training.lora
        )
        model_name = cfg.model.pretrained_base.split("/")[-1]

    model.train()

    # Create data module
    data_module = LMDataModule(cfg, tokenizer)
    data_module.prepare_data()
    data_module.setup()

    # Select the appropriate model based on training method
    if cfg.training.method == "base":
        lightning_model = BaseLMModel(model, tokenizer, cfg)
    elif cfg.training.method == "pag-hidden":
        # Fetch the index to quickly access samples given the next token
        dataset_index = DatasetIndexByToken.from_file(cfg.dataset.prefix.dataset_index_path)
        lightning_model = PAGHiddenModel(model, tokenizer, cfg, dataset_index, data_module.train_dataset)
    elif cfg.training.method == "pag-identity-embeddings":
        lightning_model = IdentityGradEmbeddingsModel(model, tokenizer, cfg)
    elif cfg.training.method == "pag-mix-identity-score-embeddings":
        lightning_model = MaskedIdentityGradEmbeddingsModel(model, tokenizer, cfg)
    elif cfg.training.method == "inv-first":
        lightning_model = InvFirstTokenModel(model, tokenizer, cfg)
    elif cfg.training.method == 'identity-grad':
        lightning_model = IdentityGradModel(model, tokenizer, cfg)
    else:
        raise ValueError(f"Unknown training method: {cfg.training.method}")

    return lightning_model, data_module, model_name


def load_model_from_checkpoint(path: pathlib.Path, device: torch.device) -> tuple[BaseLMModel, LMDataModule, str]:
    """
    Load a model from a checkpoint file.

    Args:
        path: Path to the checkpoint file.
        device: Device to load the model onto (e.g., CPU or CUDA).

    Returns:
        tuple: A tuple containing the loaded model, data module, and model name.
    """
    # If it fails to import pathlib._local,
    # it means that the checkpoint was saved with a different version of Python.
    # More specifically, on laika we have 3.13.
    # For sure, it does NOT work on 3.10.
    ckpt_data = torch.load(str(path), map_location=device, weights_only=False)

    state_dict = ckpt_data['state_dict']
    print(list(state_dict.keys()))

    hyper_parameters = ckpt_data['hyper_parameters']
    config: CustomLLMPagConfig = hyper_parameters['config']

    # Instantiate model and data module
    lightning_model, data_module, model_name = instantiate_model_by_config(config)

    lightning_model.load_state_dict(state_dict, strict=True)

    return lightning_model, data_module, model_name
