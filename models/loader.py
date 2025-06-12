"""
File with loader functions, with our hyperparameters to pass to HuggingFace transformers library.
"""
import os
import pathlib

import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedModel, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForCausalLM, LlamaConfig

from config import CustomModelConfig, DatasetConfig, LoraTConfig


def load_model_and_tokenizer(
        model_path_or_name: pathlib.Path | str,
        random_initialization: bool,
        lora_config: LoraTConfig | None = None,
) -> tuple[PreTrainedModel | nn.Module, PreTrainedTokenizerFast]:
    """
    Load the model and the tokenizer with some preconfigured parameters in the factory methods.

    Args:
        model_path_or_name (pathlib.Path | str): Path to the model directory, or the name of the pretrained.
        random_initialization (bool): Whether to load the model with random initialization, or use pretrained weights.
        lora_config (LoraTConfig | None): Configuration for LoRA (Low-Rank Adaptation) if applicable.
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


def load_tokenizer_with_config(dataset_config: DatasetConfig, model_path_or_name: pathlib.Path | str = None) -> PreTrainedTokenizerFast:
    """
    Load tokenizer based on dataset configuration or model path.
    
    Args:
        dataset_config (DatasetConfig): Dataset configuration that may contain tokenizer_name
        model_path_or_name (pathlib.Path | str, optional): Fallback model path for tokenizer loading
        
    Returns:
        PreTrainedTokenizerFast: The loaded tokenizer
    """
    if dataset_config.tokenizer_name:
        # Load external tokenizer
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            dataset_config.tokenizer_name,
            padding_side='left',
        )
        
        # Check if tokenizer has padding token, if not set it
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    elif model_path_or_name:
        # Fallback to model-based tokenizer loading
        return load_tokenizer(model_path_or_name)
    else:
        raise ValueError("Either dataset_config.tokenizer_name or model_path_or_name must be provided")


def create_model_and_tokenizer(
        dataset_cfg: DatasetConfig,
        model_cfg: CustomModelConfig,
        fast_tokenizer: PreTrainedTokenizerFast | None = None
) -> tuple[PreTrainedModel | nn.Module, PreTrainedTokenizerFast]:
    """
    Create a model and tokenizer with the specified parameters.

    Args:
        dataset_cfg (DatasetConfig): The dataset configuration.
        model_cfg (CustomModelConfig): The model configuration.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    # Check if the tokenizer already exists
    if fast_tokenizer is None:
        if not os.path.exists(f"{model_cfg.output_dir}/tokenizer-{model_cfg.vocab_size}.json"):
            # Create a tokenizer and train it
            print(f"Tokenizer not found at {model_cfg.output_dir}/tokenizer-{model_cfg.vocab_size}.json - creating it")
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            trainer = BpeTrainer(
                vocab_size=model_cfg.vocab_size,
                min_frequency=2,
                special_tokens=[
                    "<unk>",
                    "<s>",
                    "</s>",
                    "<pad>",
                    "<mask>",
                ],
            )

            # Load the dataset
            dataset = load_dataset(dataset_cfg.name, dataset_cfg.config, split="train")
            def get_training_corpus():
                for i in range(0, len(dataset), 1000):
                    yield dataset[i : i + 1000]["text"]
            training_corpus = get_training_corpus()

            # Train the tokenizer
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.train_from_iterator(
                training_corpus,
                trainer=trainer
            )

            # Save the tokenizer
            tokenizer.save(f"{model_cfg.output_dir}/tokenizer-{model_cfg.vocab_size}.json")

        # Load the tokenizer with the trained vocabulary
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{model_cfg.output_dir}/tokenizer-{model_cfg.vocab_size}.json"
        )
        fast_tokenizer.pad_token = "<pad>"
        fast_tokenizer.eos_token = "</s>"

    # Build a model
    config = LlamaConfig(
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_attention_heads=model_cfg.num_attention_heads,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_key_value_heads=model_cfg.num_key_value_heads,
        tie_word_embeddings=model_cfg.tie_word_embeddings,
        vocab_size=model_cfg.vocab_size,
        max_position_embeddings=model_cfg.max_position_embeddings,
        pad_token_id=fast_tokenizer.pad_token_id,
        bos_token_id=fast_tokenizer.bos_token_id,
        eos_token_id=fast_tokenizer.eos_token_id,
        device_map='auto',
        attn_implementation='eager',
    )
    model = AutoModelForCausalLM.from_config(config)

    return model, fast_tokenizer
