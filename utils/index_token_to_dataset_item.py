"""
Index dataset such that, given a token ID, it returns the indexes of all samples in the dataset
that have that token ID in the prefix range.
"""
import pathlib
import pickle
from collections import defaultdict
from dataclasses import dataclass

from tqdm import tqdm

from config import DatasetPrefixConfig, apply_config, Config
from data.data_processor import TextDataset, load_and_process_dataset
from models import loader


@dataclass
class DatasetSampleIndex:
    dataset_idx: int
    token_idx: int


class DatasetIndexByToken:
    def __init__(self):
        self.index_by_token: dict[int, list[DatasetSampleIndex]] = defaultdict(list)

    @staticmethod
    def from_file(path: pathlib.Path) -> 'DatasetIndexByToken':
        dataset_index_by_token = DatasetIndexByToken()
        with path.open('rb') as f:
            dataset_index_by_token.index_by_token = pickle.load(f)
        return dataset_index_by_token

    def get_samples_by_token(self, token_id: int) -> list[DatasetSampleIndex]:
        assert isinstance(token_id, int), f'token_id is not an int: {type(token_id)}'
        return self.index_by_token[token_id]

    def create_index(self, dataset: TextDataset, config: DatasetPrefixConfig):
        for dataset_idx, item in enumerate(tqdm(dataset, desc='Generating dataset index by token_id')):
            # Remember that padding is on the left, so the start and end indexes must be adjusted accordingly
            start_of_sentence = item.input_ids.size(-1) - item.attention_mask.sum().item()
            from_i, to_i = config.min_length + start_of_sentence, config.max_length + start_of_sentence

            for token_idx in range(from_i, to_i):
                token = item.input_ids[token_idx].item()
                index = DatasetSampleIndex(dataset_idx, token_idx)
                self.index_by_token[token].append(index)

                # Double check that I am accessing only legit tokens, not fillings
                assert dataset[dataset_idx].attention_mask[from_i:token_idx + 1].sum().item() \
                       == token_idx + 1 - from_i
                actual_prefix_len = dataset[dataset_idx].attention_mask[:token_idx].sum().item()
                assert config.min_length <= actual_prefix_len < config.max_length

    def save(self, out_file: pathlib.Path):
        with out_file.open('wb') as f:
            pickle.dump(self.index_by_token, f)


@apply_config()
def _main(cfg: Config):
    _, tokenizer = loader.load_model_and_tokenizer(cfg.model.pretrained_base, lora_config=None)
    train_dataset, _ = load_and_process_dataset(cfg.dataset, tokenizer, cfg.training.max_seq_length)
    index = DatasetIndexByToken()
    index.create_index(train_dataset, cfg.dataset.prefix)
    index.save(cfg.model.output_dir / 'dataset_index_by_token.pickle')

    # Use it
    token_str = ' can'
    token_id = tokenizer.batch_encode_plus([token_str], return_tensors='pt').input_ids[0, 0].item()
    print(f'{token_str=}, {token_id=}')
    retrieved = index.get_samples_by_token(token_id)
    print(f'{retrieved[0]=}')
    dataset_idx, token_idx = retrieved[0].dataset_idx, retrieved[0].token_idx

    train_dataset_sample = train_dataset[dataset_idx].input_ids[:token_idx]
    print(f'{train_dataset_sample=}, {train_dataset_sample.shape=}')
    train_dataset_sample = tokenizer.batch_decode(train_dataset_sample[None, :])[0]
    print('Train sample (Prefix): ', train_dataset_sample)
    next_token_id = train_dataset[dataset_idx].input_ids[token_idx]
    next_token_str = tokenizer.batch_decode(next_token_id[None, None])[0]
    print(f'Next token (ID: {next_token_id}): "{next_token_str}"')
    assert next_token_id == token_id
    assert next_token_str == token_str

    actual_prefix_len = train_dataset[dataset_idx].attention_mask[:token_idx].sum().item()
    assert cfg.dataset.prefix.min_length <= actual_prefix_len < cfg.dataset.prefix.max_length

    assert train_dataset[dataset_idx].attention_mask[token_idx].item() == 1


if __name__ == '__main__':
    _main()
