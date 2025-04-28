import pathlib
from collections import defaultdict

import torch
from tqdm import tqdm

from config import CustomLLMPagConfig, apply_config
from data.data_processor import BatchType
from instantiate import load_model_from_checkpoint
from models.common import compute_top_k_accuracies, forward_grad_embeddings


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig):
    device, prefix_len = 'cuda:0', 5
    torch.set_float32_matmul_precision('medium')

    lightning_module, data_module, module_name, cfg = load_model_from_checkpoint(
        pathlib.Path('./checkpoints/tinystories/tinystories_inv_first_norm__9ecoqzxt.ckpt'),
        cfg,
    )
    lightning_module.to(device)

    print(f'Loaded model: {module_name}, {type(lightning_module)}')

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
            input_ids[:, k] = lightning_module.tokenizer.pad_token_id

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
