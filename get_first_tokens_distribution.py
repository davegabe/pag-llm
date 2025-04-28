from collections import defaultdict

from matplotlib import pyplot as plt
from tqdm import tqdm

from config import CustomLLMPagConfig, apply_config
from instantiate import instantiate_model_by_config


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig):
    prefix_len = 5

    lightning_module, data_module, _ = instantiate_model_by_config(cfg)

    distributions = [defaultdict(int) for _ in range(prefix_len)]

    for batch in tqdm(data_module.val_dataloader(), desc='Computing tokens distribution'):
        for k in range(prefix_len - 1, -1, -1):
            token = batch.input_ids[:, k]
            for token_id in token:
                distributions[k][token_id.item()] += 1

    plt.figure(figsize=(10, 5))

    # Display the distribution of the first tokens (only if its frequency is at least some minimum)
    meaningful_tokens = [
        token_id
        for k_distribution in distributions
        for token_id in k_distribution.keys()
        if sum(inner_distribution[token_id] for inner_distribution in distributions) > 100
    ]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for k in range(prefix_len):
        k_distribution = distributions[k]

        # Show this distribution using matplotlib
        x_axis = [lightning_module.tokenizer.decode([token_id]) for token_id in meaningful_tokens]
        y_axis = [k_distribution[token_id] for token_id in meaningful_tokens]
        plt.bar(x_axis, y_axis, color=colors[k], label=f'K={k}')
        plt.xlabel('Token ID')
        plt.ylabel(f'Frequency at K={k}')

    plt.title(f'Distribution of First Tokens')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
