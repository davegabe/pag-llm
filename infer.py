import pathlib

import torch

import models.loader as loader
from config import LLMPagConfig, apply_config


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
    # Load model and tokenizer
    model, tokenizer = loader.load_model_and_tokenizer(model_path_or_name, lora_config=None)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

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



@apply_config()
def main(cfg: LLMPagConfig):
    # Check if checkpoints directory exists
    model_path: str | pathlib.Path
    if cfg.model.output_dir.exists():
        # Get last checkpoint
        subdirs = list(cfg.model.output_dir.glob("*"))
        subdirs = [d for d in subdirs if d.is_dir()]
        model_path = max(
            subdirs,
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