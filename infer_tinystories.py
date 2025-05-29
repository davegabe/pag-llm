import torch

import nucleus_sampling
from config import apply_config, CustomLLMPagConfig, LLMPagConfig
from instantiate import load_model_from_checkpoint
from models.base_model import BaseLMModel


def load_model(cfg: CustomLLMPagConfig | LLMPagConfig) -> BaseLMModel:
    device, prefix_len = 'cuda:0', 5
    torch.set_float32_matmul_precision('medium')

    lightning_model, data_module, module_name, cfg = load_model_from_checkpoint(
        cfg.model.output_dir / 'tinystories_bertlike_embeddings_grad_norm__sqipem6p.ckpt',
        cfg,
    )
    lightning_model.to(device)
    print(f'Loaded model: {module_name}, {type(lightning_model)}')

    # Move model to GPU if available
    lightning_model.to(device)
    lightning_model.eval()
    lightning_model.tokenizer.eos_token = '<|endoftext|>'

    return lightning_model


def generate_standard_sampling(lightning_model: BaseLMModel, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Generate text using standard sampling.
    """
    # Generate text
    outputs = lightning_model.model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=256,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # To avoid repeating text
        pad_token_id=lightning_model.tokenizer.pad_token_id,  # Explicitly set pad token id
    )
    return outputs


def generate_nucleus_sampling(lightning_model: BaseLMModel, input_ids: torch.Tensor,
                              attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Generate text using nucleus sampling.
    """
    for _ in range(20):
        logits = lightning_model.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        # Apply nucleus sampling
        next_tokens = nucleus_sampling.nucleus_sample(logits, nucleus_p=0.8, temperature=0.8)
        # Append the next token to the input_ids
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=1)

    return input_ids


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig | LLMPagConfig):
    lightning_model = load_model(cfg)

    # Example prompts
    prompts = [
        "The cat is on",
        "The pen is on",
        "Once upon a time, "
    ]

    print("-" * 80)
    for prompt in prompts:
        # Encode prompt with attention mask
        encoded_input = lightning_model.tokenizer(prompt, return_tensors="pt", padding=True).to(lightning_model.device)

        # Generate text
        outputs = generate_nucleus_sampling(lightning_model, encoded_input['input_ids'],
                                            encoded_input['attention_mask'])

        # Decode and return generated text
        generated_text = lightning_model.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Remove the text after the EOS, if present
        eos_index = generated_text.find(lightning_model.tokenizer.eos_token)
        if eos_index != -1:
            generated_text = generated_text[:eos_index]

        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
