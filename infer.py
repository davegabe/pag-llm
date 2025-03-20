import pathlib

import hydra
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
import torch

from config import Config


def generate_text(model_path_or_name: pathlib.Path | str, prompt: str, max_length: int = 50) -> str:
    """
    Generate text from a specified prompt using the model at the given path.

    Args:
        model_path_or_name (pathlib.Path | str): Path to the model directory, or the name of the pretrained.
        prompt (str): The input prompt for text generation.
        max_length (int): The maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    if isinstance(model_path_or_name, pathlib.Path):
        model_name = str(model_path_or_name.resolve())
    else:
        model_name = model_path_or_name

    # Load model and tokenizer
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Encode prompt with attention mask
    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Generate text
    outputs = model.generate(
        encoded_input["input_ids"],
        attention_mask=encoded_input["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        no_repeat_ngram_size=2,  # To avoid repeating text
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad token id
    )

    # Decode and return generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



@hydra.main(version_base=None, config_path="./config", config_name="base")
def main(cfg: Config):
    # Check if checkpoints directory exists
    model_path: str | pathlib.Path
    if cfg.model.output_dir.exists():
        # Get last checkpoint
        model_path = max(
            cfg.model.output_dir.glob("*"),
            key=lambda f: f.stat().st_mtime,
        )
        print(f"- Using fine-tuned model from {model_path}")
    else:
        # Use pre-trained model
        model_path = cfg.model.pretrained_base
        print(f"- No fine-tuned model found.")
        print(f"- Using pre-trained model from {model_path}")

    # Example prompts
    prompts = [
        "The cat is on",
        "The pen is on",
    ]

    print("-" * 80)
    for prompt in prompts:
        generated_text = generate_text(model_path, prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)


if __name__ == "__main__":
    main()