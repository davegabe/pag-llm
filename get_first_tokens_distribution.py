from math import log2

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CustomLLMPagConfig, apply_config
from instantiate import instantiate_model_by_config


def compute_tokens_distribution(train_dataloader: DataLoader, prefix_len: int, vocab_size: int) -> list[torch.Tensor]:
    distributions = [torch.zeros((vocab_size,)) for _ in range(prefix_len)]

    for batch in tqdm(train_dataloader, desc='Computing tokens distribution'):
        for k in range(prefix_len - 1, -1, -1):
            token = batch.input_ids[:, k]
            # Filter out according to the attention mask
            if prefix_len > 4:
                token = token[batch.attention_mask[:, k] == 1]
            batch_distribution = torch.bincount(token, minlength=vocab_size)
            distributions[k] += batch_distribution

    return distributions


def show_prefix_distributions(cfg: CustomLLMPagConfig, lightning_module, data_module, prefix_len: int):
    prefix_distributions = compute_tokens_distribution(data_module.train_dataloader(), prefix_len, cfg.model.vocab_size)

    # Display the distribution of the first tokens (only if its frequency is at least some minimum)
    meaningful_tokens = [
        token_id
        for token_id in range(cfg.model.vocab_size)
        if sum(inner_distribution[token_id].item() for inner_distribution in prefix_distributions) > 5_000
    ]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    colors = dict(enumerate(colors))
    max_freq = max(inner_distribution[token_id].item() for inner_distribution in prefix_distributions for token_id in
                   meaningful_tokens)

    plt.figure(figsize=(16, 6))
    plt.ylim(0, max_freq * 1.1)

    bar_width = 0.8 / prefix_len  # Adjust width so bars don't overlap

    x_labels = [lightning_module.tokenizer.decode([token_id]) for token_id in meaningful_tokens]
    x = np.arange(len(x_labels))  # the label locations

    for i in range(prefix_len):
        y_values = [prefix_distributions[i][token] for token in meaningful_tokens]
        x_offset = (i - prefix_len // 2) * bar_width + bar_width / 2
        plt.bar(x + x_offset, y_values, color=colors[i % len(colors)], label=f'Position {i}', width=bar_width)

    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of the First Tokens')
    plt.xticks(x, x_labels, rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()


def show_positions_entropy(cfg: CustomLLMPagConfig, data_module, prefix_len: int):
    # Compute the overall distribution of tokens
    entropy_positions = 100
    overall_distributions = compute_tokens_distribution(data_module.train_dataloader(),
                                                        prefix_len=entropy_positions,
                                                        vocab_size=cfg.model.vocab_size)
    # Compute the entropy at each position
    entropies = []
    for k in range(entropy_positions):
        k_distribution = overall_distributions[k]
        total_count = k_distribution.sum().item()
        entropy = -sum((count / total_count) * log2(count / total_count + 1e-4) for count in k_distribution.tolist())
        entropies.append(entropy)

    # Plot the entropy at each position
    plt.figure(figsize=(16, 6))
    plt.plot(range(prefix_len), entropies, marker='o')
    plt.xlabel('Token Position')
    plt.ylabel('Entropy')
    plt.title('Entropy of Token Distributions')
    plt.xticks(range(prefix_len), rotation=90)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig):
    cfg.training.batch_size = 4096
    lightning_module, data_module, _ = instantiate_model_by_config(cfg)

    # show_prefix_distributions(cfg, lightning_module, data_module, prefix_len=4)
    show_positions_entropy(cfg, data_module, prefix_len=100)


if __name__ == '__main__':
    main()
