import pathlib
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CustomLLMPagConfig, apply_config
from data.data_processor import BatchType
from instantiate import load_model_from_checkpoint
from models.common import compute_top_k_accuracies, forward_grad_embeddings


def load_bigram_from_file(bigram_file: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load the bigram from the file.
    :param bigram_file: Path to the bigram file.
    :return: bigram dictionary or None if the file does not exist.
    """
    if not bigram_file.exists():
        return None

    loaded_file = torch.load(str(bigram_file.resolve()), map_location='cpu')

    # Use 'unigram' for backward compatibility with an older, wrong, naming convention
    reverse_bigram = loaded_file.get('bigram') or loaded_file['unigram']
    bigram_counts = loaded_file['distribution_after_token']
    return reverse_bigram, bigram_counts


def build_and_save_bigram(train_dataloader: DataLoader, vocab_size: int, bigram_file: pathlib.Path) -> tuple[
    torch.Tensor, torch.Tensor]:
    distribution_after_token = torch.zeros((vocab_size, vocab_size),
                                           dtype=torch.int)  # [k+1 token] -> [k token] -> count

    for batch in tqdm(train_dataloader, desc='Building bigram'):
        for i_sample in range(batch.input_ids.size(0)):
            sample_len = batch.attention_mask[i_sample].sum().item()
            for k in range(sample_len - 1, 0, -1):
                # Get the k-th token
                k_token_id = batch.input_ids[i_sample, k]

                # Get the k+1-th token
                k_minus_one_token_id = batch.input_ids[i_sample, k - 1]

                # Count the occurrences of the k-th and (k-1)-th tokens
                distribution_after_token[k_token_id, k_minus_one_token_id] += 1

    # Build the bigram
    print('Building bigram...')
    # Find the most frequent token for each (k+1)-th token
    torch_bigram = torch.argmax(distribution_after_token, dim=1)

    # Save the bigram to the file
    bigram_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        'distribution_after_token': distribution_after_token,
        'bigram': torch_bigram,
    }, str(bigram_file.resolve()))

    return torch_bigram, distribution_after_token


def init_evaluation(cfg: CustomLLMPagConfig, device: str, use_init: str, ckpt_file: str):
    torch.set_float32_matmul_precision('medium')

    # Decide the strategy for the init
    allowed_init = {'bigram', 'random', 'pad'}
    assert use_init in allowed_init, \
        f'Invalid initialization strategy: {use_init}. Allowed values are: {allowed_init}'

    lightning_module, data_module, module_name, cfg = load_model_from_checkpoint(
        cfg.model.output_dir / ckpt_file,
        cfg,
    )
    lightning_module.to(device)

    model_class_name = lightning_module.__class__.__name__

    print()
    print("TESTING INVERSE LM")
    print(" - Model: ", model_class_name)
    print(" - Init strategy: ", use_init.upper())
    print()

    train_bigram_file = cfg.model.output_dir / f'train_bigram_{cfg.model.vocab_size}_full.pt'
    if train_bigram_file.exists():
        # Load the bigram from the file
        reverse_bigram, bigram_counts = load_bigram_from_file(train_bigram_file)
    else:
        # Build the bigram from the training data
        reverse_bigram, bigram_counts = build_and_save_bigram(data_module.train_dataloader(),
                                                              cfg.model.vocab_size, train_bigram_file)
    reverse_bigram, bigram_counts = map(lambda x: x.to(device), (reverse_bigram, bigram_counts))

    # To always use PAD token as the filler for the unknown token:
    if use_init == 'pad':
        reverse_bigram = torch.full_like(reverse_bigram, lightning_module.tokenizer.pad_token_id)

    return lightning_module, model_class_name, data_module, reverse_bigram, bigram_counts


def run_evaluation(device: str, prefix_len: int, use_init: str, ckpt_file: str, cfg: CustomLLMPagConfig):
    lightning_module, model_class_name, data_module, reverse_bigram, bigram_counts = init_evaluation(
        cfg=cfg,
        device=device,
        use_init=use_init,
        ckpt_file=ckpt_file,
    )

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
            ## Use the bigram
            next_token = input_ids[:, k + 1]
            input_ids[:, k] = reverse_bigram[next_token]

            ## To use a random initialization for the unknown token:
            if use_init == 'random':
                input_ids[:, k] = torch.randint_like(input_ids[:, k], 0, lightning_module.tokenizer.vocab_size)

            # Get the embeddings of X (with the k-th token replaced with [PAD])
            x_embed = lightning_module.model.get_input_embeddings()(input_ids).detach()
            x_embed.requires_grad_(True)

            outputs = lightning_module.model(
                inputs_embeds=x_embed[:, k:],
                attention_mask=attention_mask[:, k:],
                labels='dummy',
                shift_labels=shift_labels[:, k:].contiguous(),
                # Required by .view(-1) in PyTorch loss_utils.py internals
            )
            grad_x_embed = torch.autograd.grad(outputs.loss, [x_embed], create_graph=False)[0][:, k]

            # Reset the original k-th token
            input_ids[:, k] = original_k_token

            # Predict the k-th token, based on the gradients of the first token embeddings
            logits = forward_grad_embeddings(lightning_module.model, grad_x_embed)

            # Compute the top-k accuracies
            grad_top_k_accuracies = compute_top_k_accuracies(
                inv_first_label=original_k_token,
                logits=logits,
                k_samples=4,
                tag='test_inverse_lm',
            )

            # Compute the top-k accuracies of the bigram
            if use_init == 'bigram':
                bigram_logits = bigram_counts[next_token].float()
                bigram_top_k_accuracies = compute_top_k_accuracies(
                    inv_first_label=original_k_token,
                    logits=bigram_logits,
                    k_samples=4,
                    tag='test_bigram',
                )

                # Combine the accuracies to be logged later on
                top_k_accuracies = bigram_top_k_accuracies | grad_top_k_accuracies  # Python 3.9+ dict union
            else:
                top_k_accuracies = grad_top_k_accuracies

            top_k_accuracies = {k: v for k, v in top_k_accuracies.items() if '/top_' in k}
            for top_k_key, accuracy in top_k_accuracies.items():
                overall_accuracy[(top_k_key, k)] += round((accuracy * batch.input_ids.size(0)).item())

    # Compute the overall accuracy
    output_file = cfg.model.output_dir / f'inverse_lm_accuracy_{model_class_name}_{use_init}_{cfg.model.vocab_size}.txt'
    with output_file.open('w') as f:
        for top_k_key, correct_samples in overall_accuracy.items():
            top_k_accuracy = correct_samples / len(data_module.val_dataset)
            display_string = f'{top_k_key}: {top_k_accuracy:.2%}'
            print(display_string)
            f.write(display_string)


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig):
    run_evaluation(device='cuda:2',
                   prefix_len=5,
                   use_init='random',
                   ckpt_file='tinystories_identity_grad_norm__qp6q1mop.ckpt',
                   cfg=cfg)


if __name__ == '__main__':
    main()
