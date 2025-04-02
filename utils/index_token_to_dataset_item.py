"""
Index dataset such that, given a token ID, it returns the indexes of all samples in the dataset
that have that token ID in the prefix range.
"""
import logging
import multiprocessing
import pathlib
from dataclasses import dataclass

import torch
from tqdm import tqdm

from config import DatasetPrefixConfig, apply_config, Config
from data.data_module import LMDataModule
from data.data_processor import BatchType, TextDataset
from models import loader
from models.loader import load_tokenizer


@dataclass
class DatasetSampleIndex:
    dataset_idx: int
    token_idx: int


class DatasetIndexByToken:
    all_token_ids: torch.Tensor | None
    indexes_by_token: list[torch.Tensor | None] | None

    def __init__(self):
        self.logger = logging.getLogger("DatasetIndexByToken")
        self.all_token_ids = None
        self.indexes_by_token = None

    @staticmethod
    def from_file(path: pathlib.Path) -> 'DatasetIndexByToken':
        """
        Load an existing dataset indexed by token ID.

        Args:
            path: File where it got saved.

        Returns:
            DatasetIndexByToken: preloaded instance of dataset indexed by token ID.
        """
        dataset_index_by_token = DatasetIndexByToken()
        with path.open('rb') as f:
            loaded_obj = torch.load(f)
            dataset_index_by_token.indexes_by_token = loaded_obj['indexes_by_token']
            dataset_index_by_token.all_token_ids = loaded_obj['all_token_ids'].to(dtype=torch.int)
        return dataset_index_by_token

    def get_all_samples_by_token(self, token_id: int | torch.Tensor) -> torch.Tensor | None:
        """
        Find the indexes of samples and tokens, for every sample that has token_id as the next token in the prefix.

        Args:
            token_id: next token in the prefix.

        Returns:
            Tensor of shape [N, 2], where each row has (dataset_idx, token_idx).
        """
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        return self.indexes_by_token[token_id]

    def get_rand_samples_idx_by_token(self, token_id: int | torch.Tensor, num_samples: int) -> torch.Tensor | None:
        """
        Get some random samples, given the token ID to be as next token in the prefix.

        Args:
            token_id: next token in the prefix.
            num_samples: how many samples to get.

        Returns:
            Tensor of shape [num_samples, 2], where each row has (dataset_idx, token_idx).
        """
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()

        all_idx_by_token = self.get_all_samples_by_token(token_id)
        # ? What if num_samples < len(all_idx_by_token)??
        class_n = all_idx_by_token.size(0)
        if num_samples < class_n:
            rand_idx = torch.randperm(class_n, device=all_idx_by_token.device)[:num_samples]
        else:
            # Allow repetitions
            rand_idx = torch.randint(low=0, high=class_n, size=(num_samples,), device=all_idx_by_token.device)
        rand_idx_by_token = all_idx_by_token[rand_idx]
        return rand_idx_by_token

    def get_all_token_ids(self) -> torch.Tensor:
        return self.all_token_ids

    def get_rand_samples_by_token(self, dataset: TextDataset, k_classes: int, num_samples: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get some random samples, given the token ID to be as next token in the prefix.
        It will fetch the items from the dataset and create matrix batches.

        Args:
            dataset: Text Dataset to get data from
            k_classes: how many classes to get (= next tokens).
            num_samples: how many samples to get from the dataset

        Returns:
            - torch.Tensor input_ids, of shape [M * K, D]
            - torch.Tensor attention_mask of shape [M * K, D]
            - torch.Tensor next_tokens (IDs) of shape [K]
        """
        # Pick K random classes
        all_classes = self.get_all_token_ids()
        rand_classes = all_classes[torch.randperm(len(all_classes))[:k_classes]]

        all_input_ids, all_attention_masks = [], []

        for next_token_id in rand_classes:
            # Pick M random samples of that class
            pag_indexes = self.get_rand_samples_idx_by_token(next_token_id, num_samples=num_samples)
            dataset_idx, token_idx = pag_indexes[:, 0], pag_indexes[:, 1]

            pag_samples: BatchType = dataset[dataset_idx]
            input_ids, attn_mask = pag_samples.input_ids, pag_samples.attention_mask

            # Get the prefix in matrix batch form
            all_input_ids.extend([
                input_ids[i, :token_idx[i]] for i in range(len(input_ids))
            ])
            all_attention_masks.extend([
                attn_mask[i, :token_idx[i]] for i in range(len(input_ids))
            ])

        input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_side='left')
        attn_mask = torch.nn.utils.rnn.pad_sequence(all_attention_masks, batch_first=True, padding_side='left')
        ## And that's it!
        # You can pass input_ids and attn_mask to your LLM
        return input_ids, attn_mask, rand_classes

    def to(self, device: torch.device) -> 'DatasetIndexByToken':
        self.all_token_ids = self.all_token_ids.to(device=device)
        self.indexes_by_token = [
            None if tensor is None else tensor.to(device=device)
            for tensor in self.indexes_by_token
        ]
        return self

    def create_index(self, config: Config):
        # Go multithread!
        cpus = max(multiprocessing.cpu_count() - 2, 1)
        self.logger.info(f'Creating dataset prefix index, using {cpus} CPUs')

        # Download the dataset
        tokenizer = load_tokenizer(config.model.pretrained_base)
        datamodule = LMDataModule(config=config, tokenizer=tokenizer)
        datamodule.prepare_data()
        datamodule.setup()
        dataset = datamodule.train_dataset

        all_dict_files: list[pathlib.Path] = []
        all_processes = []
        chunk_size = len(dataset) // cpus
        remainder = len(dataset) % cpus

        out_dir = config.model.output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        from_i = 0
        for i in range(cpus):
            to_i = from_i + chunk_size
            if i < remainder:
                to_i += 1

            # print(from_i, to_i)
            partial_dict_file = out_dir / f'index_{i}.pt'
            all_dict_files.append(partial_dict_file)

            if not partial_dict_file.exists():
                # Run the actual process
                process = multiprocessing.Process(
                    target=DatasetIndexByToken._create_index,
                    args=(config, from_i, to_i, str(partial_dict_file)),
                )
                process.start()
                all_processes.append(process)

            # Go to the next split
            from_i = to_i

        # print(len(dataset))
        for process in all_processes:
            process.join()

        # Finally, join the results
        with torch.serialization.safe_globals([DatasetSampleIndex]):
            index_by_token: dict[int, torch.Tensor] = dict()

            for partial_dict_file in tqdm(all_dict_files, desc='Joining partial results'):
                local_dict = torch.load(str(partial_dict_file))

                for token, indexes in local_dict.items():
                    # Create the tensor with our indexes
                    new_indexes = torch.tensor([
                        [index.dataset_idx, index.token_idx]
                        for index in indexes
                    ], dtype=torch.int)

                    if token not in index_by_token:
                        index_by_token[token] = new_indexes
                    else:
                        # Resize the tensor to add our new indexes
                        tensor = index_by_token[token]
                        old_n = tensor.size(0)
                        index_by_token[token].resize_((old_n + len(indexes), 2))
                        # Write the new contents
                        index_by_token[token][old_n:] = new_indexes

        # ...and create list of tensors for faster access
        max_token_id = max(index_by_token.keys())
        self.indexes_by_token = [index_by_token.get(i, None) for i in range(max_token_id)]

        print('Creating all_token_ids tensor')
        all_tokens = sorted(index_by_token.keys())
        self.all_token_ids = torch.tensor(all_tokens, dtype=torch.uint16)

    @staticmethod
    def _create_index(app_cfg: Config, from_i: int, to_i: int, output_file: str):
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        index_by_token: dict[int, list[DatasetSampleIndex]] = dict()

        config: DatasetPrefixConfig = app_cfg.dataset.prefix
        datamodule = LMDataModule(app_cfg, load_tokenizer(app_cfg.model.pretrained_base))
        datamodule.setup()
        dataset = datamodule.train_dataset

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
                if token not in index_by_token:
                    index_by_token[token] = []
                index_by_token[token].append(index)

                # Double check that I am accessing only legit tokens, not fillings
                # assert dataset[dataset_idx].attention_mask[from_i:token_idx + 1].sum().item() \
                #        == token_idx + 1 - from_i
                # actual_prefix_len = dataset[dataset_idx].attention_mask[:token_idx].sum().item()
                # assert config.min_length <= actual_prefix_len < config.max_length

        # Save the partial result
        with open(output_file, 'wb') as f:
            torch.save(index_by_token, f)

    def save(self, out_file: pathlib.Path):
        with out_file.open('wb') as f:
            torch.save({
                'all_token_ids': self.all_token_ids,
                'indexes_by_token': self.indexes_by_token,
            }, f)


@apply_config()
def _main(cfg: Config):
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    _, tokenizer = loader.load_model_and_tokenizer(cfg.model.pretrained_base, lora_config=None)

    datamodule = LMDataModule(cfg, tokenizer)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataset = datamodule.train_dataset
    # index = DatasetIndexByToken()
    # index.create_index(config=cfg)
    # index.save(cfg.dataset.prefix.dataset_index_path)
    index = DatasetIndexByToken.from_file(cfg.dataset.prefix.dataset_index_path)

    # Use it
    k_classes = cfg.training.pag_classes
    m_samples_per_class = 3
    print(f'K = {k_classes} - M = {m_samples_per_class}')
    input_ids, attn_mask, classes = index.get_rand_samples_by_token(train_dataset, k_classes, m_samples_per_class)
    print(f'input_ids: {input_ids.shape}')
    print(f'attn_mask: {attn_mask.shape}')
    print(f'classes: {classes.shape}')

    input_ids = input_ids.view(k_classes, m_samples_per_class, -1)
    attn_mask = attn_mask.view(k_classes, m_samples_per_class, -1)

    for i, next_token_id in enumerate(classes.tolist()):
        print()
        next_token_str = tokenizer.batch_decode([[next_token_id]])[0]
        print(f'"{next_token_str}" -> {next_token_id}')

        # Pick M random samples of that class
        class_input_ids, class_attn_mask = input_ids[i], attn_mask[i]

        # Check the actual strings
        for j in range(len(class_input_ids)):
            input_id, attn = class_input_ids[j], class_attn_mask[j]
            sample_len = attn.sum().item()
            # Take last "sample_len" items
            input_id = input_id[-sample_len:]
            text = tokenizer.batch_decode([input_id])[0]
            print(f'({j}) "{text}"')


if __name__ == '__main__':
    _main()
