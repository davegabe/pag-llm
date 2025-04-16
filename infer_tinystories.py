import pathlib

import torch

import models.loader as loader
from config import apply_config, CustomLLMPagConfig, LLMPagConfig
from data.data_module import LMDataModule
from models.base_model import BaseLMModel
from models.identity_grad_embeddings_model import IdentityGradEmbeddingsModel
from models.masked_embeddings_grad_model import MaskedIdentityGradEmbeddingsModel
from models.pag_hidden_model import PAGHiddenModel
from utils.index_token_to_dataset_item import DatasetIndexByToken


def load_model(cfg: CustomLLMPagConfig | LLMPagConfig) -> BaseLMModel:
    # Check if checkpoints directory exists
    model_path: str | pathlib.Path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    model, tokenizer = loader.create_model_and_tokenizer(
        cfg.dataset,
        cfg.model
    )

    # Load the overall PyTorch Lightning module
    all_ckpt_files = cfg.model.output_dir.glob('*.ckpt')
    checkpoint_file = max(all_ckpt_files, key=lambda f: f.stat().st_mtime)
    print(f'Using checkpoint file: {checkpoint_file}')

    # Select the appropriate model based on training method
    lightning_model: BaseLMModel
    if cfg.training.method == "base":
        lightning_model = BaseLMModel.load_from_checkpoint(checkpoint_file,
                                                           model=model,
                                                           tokenizer=tokenizer,
                                                           config=cfg)
    elif cfg.training.method == "pag-hidden":
        # Fetch the index to quickly access samples given the next token
        dataset_index = DatasetIndexByToken.from_file(cfg.dataset.prefix.dataset_index_path)

        # Create data module
        data_module = LMDataModule(cfg, tokenizer)
        data_module.prepare_data()
        data_module.setup()

        lightning_model = PAGHiddenModel.load_from_checkpoint(checkpoint_file,
                                                              model=model,
                                                              tokenizer=tokenizer,
                                                              config=cfg,
                                                              dataset_index=dataset_index,
                                                              train_dataset=data_module.train_dataset)
    elif cfg.training.method == "pag-identity-embeddings":
        lightning_model = IdentityGradEmbeddingsModel.load_from_checkpoint(checkpoint_file,
                                                                           model=model,
                                                                           tokenizer=tokenizer,
                                                                           config=cfg)
    elif cfg.training.method == "pag-mix-identity-score-embeddings":
        lightning_model = MaskedIdentityGradEmbeddingsModel.load_from_checkpoint(checkpoint_file,
                                                                                 model=model,
                                                                                 tokenizer=tokenizer,
                                                                                 config=cfg)
    else:
        raise ValueError(f"Unknown training method: {cfg.training.method}")

    print('Loaded model type:', type(lightning_model))

    # Move model to GPU if available
    lightning_model.to(device)
    lightning_model.eval()
    tokenizer.eos_token = '<|endoftext|>'

    return lightning_model


@apply_config('tiny-train')
def main(cfg: CustomLLMPagConfig | LLMPagConfig):
    lightning_model = load_model(cfg)

    # Example prompts
    prompts = [
        "The cat is on",
        "The pen is on",
    ]

    print("-" * 80)
    for prompt in prompts:
        # Encode prompt with attention mask
        encoded_input = lightning_model.tokenizer(prompt, return_tensors="pt", padding=True).to(lightning_model.device)

        # Generate text
        outputs = lightning_model.model.generate(
            encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            max_length=256,
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # To avoid repeating text
            pad_token_id=lightning_model.tokenizer.pad_token_id,  # Explicitly set pad token id
        )

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
