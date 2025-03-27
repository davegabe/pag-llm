"""
File with loader functions, with our hyperparameters to pass to HuggingFace transformers library.
"""
import pathlib

from transformers import PreTrainedModel, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_path_or_name: pathlib.Path | str) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Load the model and the tokenizer with some preconfigured parameters in the factory methods.

    Args:
        model_path_or_name (pathlib.Path | str): Path to the model directory, or the name of the pretrained.
    """
    if isinstance(model_path_or_name, pathlib.Path):
        model_name = str(model_path_or_name.resolve())
    else:
        model_name = model_path_or_name

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
    )

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        model_name,
        padding_side='left',
    )

    # Check if tokenizer has padding token, if not set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
