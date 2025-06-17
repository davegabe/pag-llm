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
        pretrained_tokenizer_name: str | None = None,
) -> tuple[PreTrainedModel | nn.Module, PreTrainedTokenizerFast]:
    """
    Load the model and the tokenizer with some preconfigured parameters in the factory methods.

    Args:
        model_path_or_name (pathlib.Path | str): Path to the model directory, or the name of the pretrained.
        random_initialization (bool): Whether to load the model with random initialization, or use pretrained weights.
        lora_config (LoraTConfig | None): Configuration for LoRA (Low-Rank Adaptation) if applicable.
        pretrained_tokenizer_name (str | None): Optional name of pre-trained tokenizer to use.
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

    tokenizer = load_tokenizer(model_name, pretrained_tokenizer_name)

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


def load_tokenizer(
        model_path_or_name: pathlib.Path | str,
        pretrained_tokenizer_name: str | None = None
) -> PreTrainedTokenizerFast:
    """
    Load tokenizer from model path or use a pre-trained tokenizer.
    
    Args:
        model_path_or_name: Path to the model directory, or the name of the pretrained model.
        pretrained_tokenizer_name: Optional name of pre-trained tokenizer to use instead.
    
    Returns:
        PreTrainedTokenizerFast: The loaded tokenizer.
    """
    # Use pre-trained tokenizer if specified
    if pretrained_tokenizer_name:
        print(f"Loading pre-trained tokenizer: {pretrained_tokenizer_name}")
        tokenizer_name = pretrained_tokenizer_name
    else:
        if isinstance(model_path_or_name, pathlib.Path):
            tokenizer_name = str(model_path_or_name.resolve())
        else:
            tokenizer_name = model_path_or_name

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        tokenizer_name,
        padding_side='left',
    )

    # Check if tokenizer has padding token, if not set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def create_model_and_tokenizer(
        dataset_cfg: DatasetConfig,
        model_cfg: CustomModelConfig,
) -> tuple[PreTrainedModel | nn.Module, PreTrainedTokenizerFast]:
    """
    Create a model and tokenizer with the specified parameters.

    Args:
        dataset_cfg (DatasetConfig): The dataset configuration.
        model_cfg (CustomModelConfig): The model configuration.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    # Check if we should use a pre-trained tokenizer
    if dataset_cfg.use_pretokenized and dataset_cfg.pretrained_tokenizer_name:
        print(f"Using pre-trained tokenizer: {dataset_cfg.pretrained_tokenizer_name}")
        fast_tokenizer = AutoTokenizer.from_pretrained(
            dataset_cfg.pretrained_tokenizer_name,
            padding_side='left',
        )
            
        # Update vocab_size in model config to match tokenizer
        actual_vocab_size = len(fast_tokenizer)
        if model_cfg.vocab_size != actual_vocab_size:
            print(f"WARNING: Model vocab_size ({model_cfg.vocab_size}) doesn't match tokenizer vocab_size ({actual_vocab_size}). Using tokenizer vocab_size.")
            model_cfg.vocab_size = actual_vocab_size
            
    else:
        # Check if the tokenizer already exists
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
                    "<|endoftext|>" # Since TinyStoriesV2_cleaned dataset has this token
                ],
            )

            # Load the dataset
            dataset = load_dataset(dataset_cfg.name, split="train")
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
