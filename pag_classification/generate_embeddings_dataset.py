import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, PreTrainedModel

import models.loader as loader
from config import apply_config, SentenceClassificationConfig
from pag_classification.embeddings_datamodule import SentenceEmbeddingsDataModule


@torch.no_grad()
def embed_dataset(dataset: torch.utils.data.Dataset, tokenizer: PreTrainedTokenizerFast,
                  embedder_model: PreTrainedModel) -> dict:
    dataloader = DataLoader(dataset, batch_size=64)

    sentence_to_embedding = dict()

    for batch in tqdm(dataloader):
        batch_x, batch_y = batch['text'], batch['label']

        # Tokenize the batch
        batch_tokenized = tokenizer(batch_x, padding=True, truncation=False, return_tensors='pt') \
            .to(embedder_model.device)

        # Do a forward pass to get the embeddings
        pass_result = embedder_model(
            **batch_tokenized,
            output_hidden_states=True,
            return_dict=True,
        )

        # The embeddings are the hidden states at the last layer, for the [CLS] token
        batch_embeddings = pass_result.hidden_states[-1][:, 0, :].cpu()  # [B, D]

        for i, (sentence, label) in enumerate(zip(batch_x, batch_y)):
            sentence_to_embedding[sentence] = {
                'text': sentence,
                'embedding': batch_embeddings[i],
                'label': label.item(),
            }

    return sentence_to_embedding


def remove_too_long_sentences(full_dataset, max_length: int) -> torch.utils.data.Dataset:
    """
    Filter out sentences longer than `max_length`, which allows them to fit inside the model context window.
    """
    short_enough_idx = [
        idx
        for idx, text in enumerate(full_dataset['text'])
        if len(text) < max_length
    ]

    print('Extracted', len(short_enough_idx), 'sentences, out of', len(full_dataset))

    return torch.utils.data.Subset(full_dataset, short_enough_idx)


@apply_config('sentence_classification')
def main(cfg: SentenceClassificationConfig):
    model, tokenizer = loader.load_model_and_tokenizer(
        cfg.backbone_model,
        random_initialization=False,
    )

    # Load the full dataset
    full_sentences_dataset = load_dataset(cfg.sentences_dataset).with_format('torch')

    # Remove too long sentences, which should be truncated from the model context window
    sentence_max_length = 1200
    sentences_train_dataset = remove_too_long_sentences(full_sentences_dataset['train'], sentence_max_length)
    sentences_test_dataset = remove_too_long_sentences(full_sentences_dataset['test'], sentence_max_length)

    # Embed the train split
    train_sentence_to_embedding = embed_dataset(sentences_train_dataset, tokenizer, model)
    train_file = SentenceEmbeddingsDataModule.get_train_split_embeddings_file(cfg)
    train_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_sentence_to_embedding, train_file)

    # Embed the test split
    test_sentence_to_embedding = embed_dataset(sentences_test_dataset, tokenizer, model)
    test_file = SentenceEmbeddingsDataModule.get_test_split_embeddings_file(cfg)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_sentence_to_embedding, test_file)
