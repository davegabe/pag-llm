"""
File with loader functions, with our hyperparameters to pass to HuggingFace transformers library.
"""
import pathlib

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForCausalLM

from config import LoraTConfig


def load_model_and_tokenizer(
        model_path_or_name: pathlib.Path | str,
        random_initialization: bool,
        lora_config: LoraTConfig | None = None,
) -> tuple[PreTrainedModel | nn.Module, PreTrainedTokenizerFast]:
    """
    Load the model and the tokenizer with some preconfigured parameters in the factory methods.

    Args:
        model_path_or_name (pathlib.Path | str): Path to the model directory, or the name of the pretrained.
        lora_config
    """
    if isinstance(model_path_or_name, pathlib.Path):
        model_name = str(model_path_or_name.resolve())
    else:
        model_name = model_path_or_name

    # Load the model and tokenizer
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        attn_implementation='eager',
    )

    tokenizer = load_tokenizer(model_name)

    # Load the model with random initialization
    if random_initialization:
        print(f"WARN - Loading model with random initialization: {model_name}")
        model: PreTrainedModel = AutoModelForCausalLM.from_config(model.config)

    # Set the model configuration
    if lora_config and lora_config.use_lora:
        lora_config = LoraConfig(
            r=lora_config.lora_rank,
            lora_alpha=lora_config.lora_rank*2, # As used in (https://github.com/microsoft/LoRA)
            lora_dropout=lora_config.lora_dropout
        )
        peft_model = get_peft_model(model, lora_config)
        return peft_model, tokenizer
    else:
        return model, tokenizer


def load_tokenizer(model_path_or_name: pathlib.Path | str) -> PreTrainedTokenizerFast:
    if isinstance(model_path_or_name, pathlib.Path):
        model_name = str(model_path_or_name.resolve())
    else:
        model_name = model_path_or_name

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left',
    )

    # Check if tokenizer has padding token, if not set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer