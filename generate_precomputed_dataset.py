"""
File to generate a precomputed dataset of hidden states.
"""
import torch
from tqdm import tqdm

import loader
from config import Config, apply_config
from data_processor import load_and_process_dataloader


@apply_config()
def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = loader.load_model_and_tokenizer(cfg.model.pretrained_base)

    train_dataloader, _ = load_and_process_dataloader(cfg.dataset, cfg.training, tokenizer, cfg.training.max_seq_length)

    # Take every token between prefix min and max length
    prefix_tokens = set()
    for batch in tqdm(train_dataloader):
        input_ids, attn_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        batch_size = input_ids.size(0)

        sentences_start_idx: torch.Tensor = cfg.training.max_seq_length - attn_mask.sum(dim=1)
        prefix_len = cfg.dataset.prefix.max_length - cfg.dataset.prefix.min_length
        # To get the prefix tokens using batched operations:
        # - Start with the index of the start token of each sentence (sentences_start_idx)
        # - Add the minimum index to start taking prefixes
        # - Add the shift per token requested, the same for every sentence
        # - Add the shift to have a flattened index to pass to torch.take()
        flat_start_ids = sentences_start_idx[:, None] + \
                         cfg.dataset.prefix.min_length + \
                         torch.arange(prefix_len, device=device).repeat(batch_size).view(batch_size, -1) + \
                         torch.arange(batch_size, device=device)[:, None] * cfg.training.max_seq_length

        prefixes = torch.take(input_ids, flat_start_ids)

        # text_prefixes = tokenizer.batch_decode(prefixes, skip_special_tokens=True)
        # print(text_prefixes)

        prefix_tokens.update(torch.flatten(prefixes).tolist())

    # train_data_batch = next(iter(train_dataloader))
    # train_data_batch.to(device)
    # print(train_data_batch)
    #
    # outputs: GenerateDecoderOnlyOutput = model.generate(
    #     **train_data_batch,
    #     generation_config=GenerationConfig(
    #         output_hidden_states=True,
    #         return_dict_in_generate=True,
    #         max_new_tokens=10,
    #     ),
    # )
    #
    # selected_layer_hidden_state = outputs.hidden_states[0][cfg.model.hidden_layer_index]
    # print('Input tokens:', train_data_batch['input_ids'].shape, '-', 'Individual lengths:', train_data_batch['attention_mask'].sum(dim=1))
    # print('Hidden states:', selected_layer_hidden_state.shape)
    #
    # output_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    # print(output_text)


if __name__ == "__main__":
    main()
