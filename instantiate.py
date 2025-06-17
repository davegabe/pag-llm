import pathlib

import torch

import models.loader as loader
from config import LLMPagConfig, CustomLLMPagConfig
from data.data_module import LMDataModule
from models.base_model import BaseLMModel
from models.identity_grad_embeddings_model import IdentityGradEmbeddingsModel
from models.inv_first_token import InvFirstTokenModel
from models.masked_embeddings_grad_model import MaskedIdentityGradEmbeddingsModel


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
        model_name = None  # Use the model_name parameter in BaseLMModel
    else:
        model, tokenizer = loader.load_model_and_tokenizer(
            cfg.model.pretrained_base,
            cfg.model.random_initialization,
            cfg.training.lora,
            cfg.dataset.pretrained_tokenizer_name if cfg.dataset.use_pretokenized else None
        )
        model_name = cfg.model.pretrained_base.split("/")[-1]

    model.train()

    # Create data module
    data_module = LMDataModule(cfg, tokenizer)
    data_module.prepare_data()
    data_module.setup()

    # Select the appropriate model based on training method
    # Some of them have multiple naming conventions, because of old, convoluted, names
    # left in the saved configuration in the corresponding .ckpt files.
    if cfg.training.method == "base":
        lightning_model = BaseLMModel('base', model, tokenizer, cfg)
    elif cfg.training.method in ("bert-like", "pag-mix-identity-score-embeddings"):
        lightning_model = MaskedIdentityGradEmbeddingsModel(model, tokenizer, cfg)
    elif cfg.training.method == "inv-first":
        lightning_model = InvFirstTokenModel(model, tokenizer, cfg)
    elif cfg.training.method in ("identity-grad", "pag-identity-embeddings"):
        lightning_model = IdentityGradEmbeddingsModel(model, tokenizer, cfg)
    else:
        raise ValueError(f"Unknown training method: {cfg.training.method}")

    return lightning_model, data_module, model_name or lightning_model.model_name


def load_model_from_checkpoint(path: pathlib.Path, current_cfg: CustomLLMPagConfig) -> tuple[
    BaseLMModel, LMDataModule, str, CustomLLMPagConfig]:
    """
    Load a model from a checkpoint file.

    Args:
        path: Path to the checkpoint file.
        current_cfg: Current configuration object to get some parameters from the environment.

    Returns:
        tuple: A tuple containing the loaded model, data module, model name, and configuration it was trained on.
    """
    # If it fails to import pathlib._local,
    # it means that the checkpoint was saved with a different version of Python.
    # More specifically, on laika we have 3.13.
    # For sure, it does NOT work on 3.10 (Cineca) neither on 3.12 (DI cluster).
    torch_device = 'cpu'
    if torch.cuda.is_available():
        torch_device = 'cuda'
    if current_cfg.training.device is not None:
        torch_device = f'cuda:{current_cfg.training.device[0]}'
    ckpt_data = torch.load(str(path), map_location=torch_device, weights_only=False)

    state_dict = ckpt_data['state_dict']

    hyper_parameters = ckpt_data['hyper_parameters']
    config: CustomLLMPagConfig = hyper_parameters['config']

    # Double check that the tokenizer JSON already exists
    vocab_size = config.model.vocab_size
    vocab_json_file = config.model.output_dir / f'tokenizer-{vocab_size}.json'
    assert vocab_json_file.exists(), f"Tokenizer JSON file not found: {vocab_json_file}"

    # Update some parameters in the config from the environment
    config.training.device = current_cfg.training.device
    config.training.batch_size = current_cfg.training.batch_size
    config.training.run_evaluation_before_training = current_cfg.training.run_evaluation_before_training

    # Instantiate model and data module
    lightning_model, data_module, model_name = instantiate_model_by_config(config)

    lightning_model.load_state_dict(state_dict, strict=True)

    return lightning_model, data_module, model_name, config
