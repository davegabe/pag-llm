import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from config import MODEL_CHECKPOINT, OUTPUT_DIR


def generate_text(model_path: str, prompt: str, max_length: int = 50) -> str:
    """
    Generate text from a specified prompt using the model at the given path.

    Args:
        model_path (str): Path to the model directory.
        prompt (str): The input prompt for text generation.
        max_length (int): The maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path
    )

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


if __name__ == "__main__":
    # Check if checkpoints directory exists
    if os.path.exists(OUTPUT_DIR):
        # Get last checkpoint
        model_path = max(
            [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)],
            key=os.path.getmtime,
        )
        print(f"- Using fine-tuned model from {model_path}")
    else:
        # Use pre-trained model
        model_path = MODEL_CHECKPOINT
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
