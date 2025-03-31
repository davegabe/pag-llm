"""
Index dataset such that, given a token ID, it returns the indexes of all samples in the dataset
that have that token ID in the prefix range.
"""
import logging
import multiprocessing
import pathlib
import pickle
from collections import defaultdict
from dataclasses import dataclass

from tqdm import tqdm

from config import DatasetPrefixConfig, apply_config, Config
from data.data_module import LMDataModule
from data.data_processor import TextDataset
from models import loader


@dataclass
class DatasetSampleIndex:
    dataset_idx: int
    token_idx: int


DatasetIndexDict = dict[int, list[DatasetSampleIndex]]


class DatasetIndexByToken:
    def __init__(self):
        self.index_by_token: DatasetIndexDict = defaultdict(list)
        self.logger = logging.getLogger("DatasetIndexByToken")

    @staticmethod
    def from_file(path: pathlib.Path) -> 'DatasetIndexByToken':
        dataset_index_by_token = DatasetIndexByToken()
        with path.open('rb') as f:
            dataset_index_by_token.index_by_token = pickle.load(f)
        return dataset_index_by_token

    def get_samples_by_token(self, token_id: int) -> list[DatasetSampleIndex]:
        assert isinstance(token_id, int), f'token_id is not an int: {type(token_id)}'
        return self.index_by_token[token_id]

    def get_all_token_ids(self) -> list[int]:
        return list(self.index_by_token.keys())

    def create_index(self, dataset: TextDataset, config: DatasetPrefixConfig):
        # Go multithread!
        cpus = max(multiprocessing.cpu_count() - 3, 1)
        self.logger.info(f'Creating dataset prefix index, using {cpus} CPUs')

        manager = multiprocessing.Manager()
        all_dicts = []
        all_processes = []
        chunk_size = len(dataset) // cpus
        remainder = len(dataset) % cpus

        from_i = 0
        for i in range(cpus):
            to_i = from_i + chunk_size
            if i < remainder:
                to_i += 1

            # print(from_i, to_i)
            local_dict = manager.dict()
            all_dicts.append(local_dict)

            # Run the actual process
            process = multiprocessing.Process(
                target=DatasetIndexByToken._create_index,
                args=(dataset, config, from_i, to_i, local_dict),
            )
            process.start()
            all_processes.append(process)

            from_i = to_i

        # print(len(dataset))
        for process in all_processes:
            process.join()

        # Finally, join the results
        for local_dict in all_dicts:
            for token, indexes in local_dict.items():
                self.index_by_token[token].extend(indexes)

    @staticmethod
    def _create_index(dataset: TextDataset, config: DatasetPrefixConfig, from_i: int, to_i: int,
                      index_by_token: DatasetIndexDict):
        print(f'Running job from {from_i} to {to_i}')

        dataset_iter = range(from_i, to_i)
        # Show the progress bar only for the #0 thread
        if from_i == 0:
            dataset_iter = tqdm(dataset_iter, desc='Generating dataset index by token_id')

        for dataset_idx in dataset_iter:
            item = dataset[dataset_idx]
            # Remember that padding is on the left, so the start and end indexes must be adjusted accordingly
            start_of_sentence = item.input_ids.size(-1) - item.attention_mask.sum().item()
            from_i, to_i = config.min_length + start_of_sentence, config.max_length + start_of_sentence

            for token_idx in range(from_i, to_i):
                token = item.input_ids[token_idx].item()
                index = DatasetSampleIndex(dataset_idx, token_idx)

                prev_indexes: list[DatasetSampleIndex]
                if token in index_by_token:
                    prev_indexes = index_by_token[token]
                else:
                    prev_indexes = []
                prev_indexes.append(index)
                index_by_token[token] = prev_indexes  # Required by multiprocessing

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
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    _, tokenizer = loader.load_model_and_tokenizer(cfg.model.pretrained_base, lora_config=None)

    datamodule = LMDataModule(cfg, tokenizer)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataset = datamodule.train_dataset
    print(train_dataset[148].input_ids[500:])
    index = DatasetIndexByToken()
    index.create_index(dataset=train_dataset, config=cfg.dataset.prefix)
    index.save(cfg.model.output_dir / 'dataset_index_by_token.pickle')

    # Use it
    token_id = index.get_all_token_ids()[32]
    token_str = tokenizer.batch_decode([[token_id]])[0]
    token_id = tokenizer.batch_encode_plus([token_str], return_tensors='pt').input_ids[0, 0].item()
    print(f'{token_str=}, {token_id=}')
    retrieved = index.get_samples_by_token(token_id)
    print(f'Retrieved {len(retrieved)} samples')
    print(f'{retrieved[0]=}')
    dataset_idx, token_idx = retrieved[0].dataset_idx, retrieved[0].token_idx

    train_dataset_sample = train_dataset[dataset_idx].input_ids[:token_idx]
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
