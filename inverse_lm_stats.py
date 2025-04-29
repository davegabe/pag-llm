import pathlib
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CustomLLMPagConfig, apply_config
from data.data_processor import BatchType
from instantiate import load_model_from_checkpoint
from models.common import compute_top_k_accuracies, forward_grad_embeddings


def load_unigram_from_file(unigram_file: pathlib.Path) -> torch.Tensor | None:
    """
    Load the unigram from the file.
    :param unigram_file: Path to the unigram file.
    :return: Unigram dictionary or None if the file does not exist.
    """
    if not unigram_file.exists():
        return None

    return torch.load(str(unigram_file.resolve()), map_location='cpu')['unigram']


def build_and_save_unigram(train_dataloader: DataLoader, vocab_size: int, unigram_file: pathlib.Path) -> torch.Tensor:
    distribution_after_token = torch.zeros((vocab_size, vocab_size),
                                           dtype=torch.int)  # [k+1 token] -> [k token] -> count

    for batch in tqdm(train_dataloader, desc='Building unigram'):
        for i_sample in range(batch.input_ids.size(0)):
            sample_len = batch.attention_mask[i_sample].sum().item()
            for k in range(sample_len - 1, 0, -1):
                # Get the k-th token
                k_token_id = batch.input_ids[i_sample, k]

                # Get the k+1-th token
                k_minus_one_token_id = batch.input_ids[i_sample, k - 1]

                # Count the occurrences of the k-th and (k-1)-th tokens
                distribution_after_token[k_token_id, k_minus_one_token_id] += 1

    # Build the unigram
    print('Building unigram...')
    # Find the most frequent token for each (k+1)-th token
    torch_unigram = torch.argmax(distribution_after_token, dim=1)

    # Save the unigram to the file
    unigram_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        'distribution_after_token': distribution_after_token,
        'unigram': torch_unigram,
    }, str(unigram_file.resolve()))

    return torch_unigram


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig):
    device, prefix_len = 'cuda:3', 5
    torch.set_float32_matmul_precision('medium')

    lightning_module, data_module, module_name, cfg = load_model_from_checkpoint(
        cfg.model.output_dir / 'tinystories_identity_grad_norm__qp6q1mop.ckpt',
        cfg,
    )
    lightning_module.to(device)
    print(f'Loaded model: {module_name}, {type(lightning_module)}')

    train_unigram_file = cfg.model.output_dir / f'train_unigram_{cfg.model.vocab_size}_full.pt'
    if train_unigram_file.exists():
        # Load the unigram from the file
        reverse_unigram = load_unigram_from_file(train_unigram_file)
    else:
        # Build the unigram from the training data
        reverse_unigram = build_and_save_unigram(data_module.train_dataloader(), cfg.model.vocab_size,
                                                 train_unigram_file)
    reverse_unigram = reverse_unigram.to(device)

    ## To always use PAD token as the filler for the unknown token,
    ## Uncomment the following line:
    # reverse_unigram = torch.full_like(reverse_unigram, lightning_module.tokenizer.pad_token_id)

    # Do a forward testing
    # trainer = Trainer(devices='0,')
    # trainer.validate(lightning_module, data_module.test_dataloader())

    overall_accuracy = defaultdict(int)
    lightning_module.eval()

    for batch in tqdm(data_module.val_dataloader(), desc='Inverse LM evaluation'):
        batch: BatchType = batch.to(torch.device(device))
        input_ids, attention_mask, shift_labels = batch.input_ids, batch.attention_mask, batch.shift_labels

        for k in range(prefix_len - 1, -1, -1):
            # Do a forward pass,
            # letting the model see only the tokens after k-th.
            # The model must predict token k-th
            # To do that, we need to mask the k-th token with [PAD]
            original_k_token = input_ids[:, k].clone()

            # Replace the k-th token with [PAD]
            # input_ids[:, k] = lightning_module.tokenizer.pad_token_id
            ## Use the unigram
            input_ids[:, k] = reverse_unigram[input_ids[:, k + 1]]

            # Get the embeddings of X (with the k-th token replaced with [PAD])
            x_embed = lightning_module.model.get_input_embeddings()(input_ids).detach()
            x_embed.requires_grad_(True)

            outputs = lightning_module.model(
                inputs_embeds=x_embed[:, k:],
                attention_mask=attention_mask[:, k:],
                labels='dummy',
                shift_labels=shift_labels[:, k:].clone(),  # Required by .view(-1) in PyTorch loss_utils.py internals
            )
            grad_x_embed = torch.autograd.grad(outputs.loss, [x_embed], create_graph=False)[0][:, k]

            # Reset the original k-th token
            input_ids[:, k] = original_k_token

            # Predict the k-th token, based on the gradients of the first token embeddings
            logits = forward_grad_embeddings(lightning_module.model, grad_x_embed)

            # Compute the top-k accuracies
            top_k_accuracies = compute_top_k_accuracies(
                inv_first_label=original_k_token,
                logits=logits,
                k_samples=4,
                tag='test',
            )
            top_k_accuracies = {k: v for k, v in top_k_accuracies.items() if k.startswith('test/top_')}

            for top_k_key, accuracy in top_k_accuracies.items():
                overall_accuracy[(top_k_key, k)] += round(accuracy.item() * batch.input_ids.size(0))

    # Compute the overall accuracy
    for top_k_key, correct_samples in overall_accuracy.items():
        top_k_accuracy = correct_samples / len(data_module.val_dataset)
        print(f'{top_k_key}: {top_k_accuracy:.2%}')


if __name__ == '__main__':
    main()
