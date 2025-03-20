"""
File to generate a precomputed dataset of hidden states.
"""
import torch
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput

import loader
from config import Config, apply_config
from data_processor import load_and_process_dataloader, load_and_process_dataset


@apply_config()
def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = loader.load_model_and_tokenizer(cfg.model.pretrained_base)

    train_dataloader, _ = load_and_process_dataloader(cfg.dataset, cfg.training, tokenizer, cfg.training.max_seq_length)

    train_data_batch = next(iter(train_dataloader))
    train_data_batch.to(device)
    print(train_data_batch)

    outputs: GenerateDecoderOnlyOutput = model.generate(
        **train_data_batch,
        generation_config=GenerationConfig(
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=10,
        ),
    )

    selected_layer_hidden_state = outputs.hidden_states[0][cfg.model.hidden_layer_index]
    print('Input tokens:', train_data_batch['input_ids'].shape, '-', 'Individual lengths:', train_data_batch['attention_mask'].sum(dim=1))
    print('Hidden states:', selected_layer_hidden_state.shape)

    output_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    print(output_text)


if __name__ == "__main__":
    main()
