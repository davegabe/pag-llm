"""
File to generate a precomputed dataset of hidden states.
"""
from functools import reduce

import torch
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput

import loader
from config import Config, apply_config
from data_processor import load_and_process_dataloader


@apply_config()
@torch.no_grad()
def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = loader.load_model_and_tokenizer(cfg.model.pretrained_base)

    train_dataloader, _ = load_and_process_dataloader(cfg.dataset, cfg.training, tokenizer, cfg.training.max_seq_length)

    # Take every token between prefix min and max length
    for batch in tqdm(train_dataloader):
        input_ids, attn_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        batch_size = input_ids.size(0)

        sentences_start_idx: torch.Tensor = cfg.training.max_seq_length - attn_mask.sum(dim=1)
        prefix_len = cfg.dataset.prefix.max_length  # Include also the tokens before the min_length = the context
        # To get the prefix tokens using batched operations:
        # - Start with the index of the start token of each sentence (sentences_start_idx)
        # - Add the shift per token requested, the same for every sentence
        # - Add the shift to have a flattened index to pass to torch.take()
        flat_start_ids = sentences_start_idx[:, None] + \
                         torch.arange(prefix_len, device=device).repeat(batch_size).view(batch_size, -1) + \
                         torch.arange(batch_size, device=device)[:, None] * cfg.training.max_seq_length

        prefixes = torch.take(input_ids, flat_start_ids)

        # Double check that the attention masks are all ones
        attn_mask_items = torch.take(attn_mask, flat_start_ids).sum().cpu().item()
        prefixes_items_count = reduce(lambda x, y: x * y, prefixes.shape, 1)
        assert attn_mask_items == prefixes_items_count

        outputs: GenerateDecoderOnlyOutput = model.generate(
            input_ids=prefixes,
            attention_mask=torch.ones_like(prefixes),
            generation_config=GenerationConfig(
                max_new_tokens=1,  # We only need the hidden states for the prefix tokens
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )

        selected_layer_hidden_state = outputs.hidden_states[0][cfg.model.hidden_layer_index]
        # Now we have to ignore the first min_length tokens, since they are only context, not our desired X
        prefix_hidden_state = selected_layer_hidden_state[:, cfg.dataset.prefix.min_length:, :]
        tqdm.write(f'Hidden states: {prefix_hidden_state.shape}\n')


if __name__ == "__main__":
    main()
