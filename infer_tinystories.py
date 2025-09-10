import sys
import torch

from config import apply_config, CustomLLMPagConfig, LLMPagConfig
from instantiate import load_model_from_checkpoint
from models.base_model import BaseLMModel


def load_model(cfg: CustomLLMPagConfig | LLMPagConfig) -> BaseLMModel:
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = cfg.training.device or default_device
    torch.set_float32_matmul_precision('medium')

    # Load model and data
    lightning_model, data_module, module_name, cfg = load_model_from_checkpoint(
        cfg.model.checkpoint_path,
        cfg,
    )
    lightning_model.to(device)
    print(f'Loaded model: {module_name}, {type(lightning_model)}', file=sys.stderr)

    # Move model to GPU if available
    lightning_model.to(device)
    lightning_model.eval()
    lightning_model.tokenizer.eos_token = '<|endoftext|>'

    return lightning_model


@apply_config('tiny-train')
def main(cfg: CustomLLMPagConfig | LLMPagConfig):
    lightning_model = load_model(cfg)

    # Example prompts
    prompts = [
        #"The cat is on",
        #"The pen is on",
        #"Once upon a time, ",
        #"Once there was a",
        "One day,"
    ]

    print("-" * 80)
    for prompt in prompts:
        # Encode prompt with attention mask
        encoded_input = lightning_model.tokenizer(prompt, return_tensors="pt", padding=True).to(lightning_model.device)

        # Generate text
        outputs = lightning_model.model.generate(
            encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            max_length=40,
            num_return_sequences=1,
            do_sample=True,
            top_p=3,
            top_k=5,
            pad_token_id=lightning_model.tokenizer.pad_token_id,  # Explicitly set pad token id
        )

        # Decode and return generated text
        generated_text = lightning_model.tokenizer \
            .decode(outputs[0], skip_special_tokens=True) \
            .replace(' ', '') \
            .replace('▁', ' ')[1:]  # Remove leading space added by tokenizer

        # Remove the text after the EOS, if present
        eos_index = generated_text.find(lightning_model.tokenizer.eos_token)
        if eos_index != -1:
            generated_text = generated_text[:eos_index]

        # Remove the prefix prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
