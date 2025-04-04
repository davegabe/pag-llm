import pathlib

import torch
from torch.utils.data import Dataset


class SentenceEmbeddingsDataset(Dataset):
    def __init__(self, embeddings_file: pathlib.Path):
        self.sentence_to_embedding_dict = torch.load(str(embeddings_file.resolve()))
        self.sentences = list(self.sentence_to_embedding_dict.keys())

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return self.sentence_to_embedding_dict[sentence]
