import json
import pathlib

import torch
import torch.nn.functional as F

from config import CustomLLMPagConfig, apply_config
from data.data_processor import TextDataset
from gcg import gcg_algorithm, gcg_evaluation
from instantiate import load_model_from_checkpoint
from models.base_model import BaseLMModel


def run_gcg_single_attack(gcg: gcg_algorithm.GCG, target_response: str):
    x_attack_str, y_attack_response, _ = gcg.run(target_response,
                                                 evaluate_every_n_steps=50,
                                                 stop_after_same_loss_steps=10)
    print(f"Attack string: {x_attack_str}")
    print(f"Attack response: {y_attack_response}")
    print(f"Desired response: {target_response}")


def run_full_gcg_evaluation(gcg: gcg_algorithm.GCG, dataset: TextDataset, gcg_output_file: pathlib.Path):
    print('Attacking:', gcg_output_file.stem, 'on', gcg.device)
    gcg_results = gcg_evaluation.evaluate_model_with_gcg(gcg, dataset,
                                                         target_response_len=10,
                                                         max_samples_to_attack=1_000,
                                                         random_select_samples=False)
    with gcg_output_file.open('w') as f:
        json.dump([r.to_dict() for r in gcg_results], f, indent=4)
    print(f"Saved GCG results to {gcg_output_file}")


def analyze_gcg_results(lightning_model: BaseLMModel, gcg_output_file: pathlib.Path, batch_size: int = 128):
    with gcg_output_file.open('r') as f:
        gcg_results = [gcg_evaluation.GCGResult.from_dict(r) for r in json.load(f)]

    # Count the number of successfully attacked tokens
    num_successful_tokens = sum(
        result.get_success_tokens()
        for result in gcg_results
    )
    num_total_tokens = sum(
        min(len(result.target_response_ids), len(result.y_attack_response_ids))
        for result in gcg_results
    )
    token_attack_success_rate = num_successful_tokens / num_total_tokens if num_total_tokens > 0 else 0
    print(f'Token attack success rate: {token_attack_success_rate:.2%}')

    # Count the average required steps to converge
    # Ignore attacks with less than 2 successful tokens
    successful_steps = [
        result.steps
        for result in gcg_results
        if result.get_success_tokens() > 1
    ]
    mean_success_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0
    stddev_success_steps = (sum((x - mean_success_steps) ** 2 for x in successful_steps) / len(
        successful_steps)) ** 0.5 if successful_steps else 0
    print(f'Mean steps to success: {mean_success_steps:.0f} ± {stddev_success_steps:.0f}')

    # Compute the KL-divergence and CE-loss of the target suffix, given both the attack and the original prefix
    tokenizer, llm = lightning_model.tokenizer, lightning_model.model
    overall_original_loss, overall_attack_loss, overall_kl_div = [], [], []
    for start_i in range(0, len(gcg_results), batch_size):
        end_i = min(start_i + batch_size, len(gcg_results))
        batch = gcg_results[start_i:end_i]

        original_prefix_ids = torch.tensor([result.original_prefix_ids for result in batch], device=llm.device)
        target_response_ids = torch.tensor([result.target_response_ids for result in batch], device=llm.device)
        y_attack_response_ids = torch.tensor([result.y_attack_response_ids for result in batch], device=llm.device)
        x_attack_ids = tokenizer.batch_encode_plus([result.x_attack_str for result in batch],
                                                   return_tensors='pt')['input_ids'].to(llm.device)

        prefix_len = original_prefix_ids.size(1)
        batch_size, suffix_len = y_attack_response_ids.shape
        vocab_size = tokenizer.vocab_size

        @torch.no_grad()
        def _compute_y_logits(x: torch.Tensor) -> torch.Tensor:
            """
            Compute the logits that correspond to the prediction of the target suffix,
            given a prefix which may be the original prefix or the attack prefix.

            Args:
                x: The prefix tokens to be used for the forward pass.

            Returns:
                The logits for the target suffix, already flattened.
            """
            nonlocal llm, target_response_ids, prefix_len
            # Compute the forward pass with the given prefix
            llm_tokens = torch.cat([x, target_response_ids], dim=1)
            logits = llm(llm_tokens, return_dict=True).logits

            # Extract only the logits that should predict the target response
            return logits[:, prefix_len - 1:-1, :]

        def _compute_logits_ce_loss(logits: torch.Tensor) -> torch.Tensor:
            nonlocal target_response_ids, vocab_size, batch_size, suffix_len
            # Compute the cross-entropy loss for the every sample in the batch
            return F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_response_ids.view(-1),
                reduction='none',
            ).view(batch_size, suffix_len).mean(dim=-1)

        with torch.no_grad():
            # Compute the forward pass for the original and attack prefixes
            original_logits = _compute_y_logits(original_prefix_ids)
            attack_logits = _compute_y_logits(x_attack_ids)

            # Compute the CE-loss for both
            overall_original_loss.append(_compute_logits_ce_loss(original_logits))
            overall_attack_loss.append(_compute_logits_ce_loss(attack_logits))

            # Compute the KL-divergence between the two
            kl_div = F.kl_div(
                attack_logits.reshape(-1, vocab_size).log_softmax(dim=-1),
                original_logits.reshape(-1, vocab_size).log_softmax(dim=-1),
                reduction='none',
                log_target=True,
            ).sum(dim=-1).view(batch_size, suffix_len).mean(dim=-1)
            overall_kl_div.append(kl_div)

    # Compute the mean of the losses
    original_loss = torch.cat(overall_original_loss)
    original_loss_mean, original_loss_stddev = original_loss.mean(), original_loss.std()
    print(f'Mean original X CE-loss: {original_loss_mean:.2f} ± {original_loss_stddev:.2f}')

    attack_loss = torch.cat(overall_attack_loss)
    attack_loss_mean, attack_loss_stddev = attack_loss.mean(), attack_loss.std()
    print(f'Mean attack X CE-loss: {attack_loss_mean:.2f} ± {attack_loss_stddev:.2f}')

    kl_div = torch.cat(overall_kl_div)
    kl_div_mean, kl_div_stddev = kl_div.mean(), kl_div.std()
    print(f'Mean KL-divergence between original and attack Xs: {kl_div_mean:.2f} ± {kl_div_stddev:.2f}')


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig):
    """
    Main function to train the model with the Inverse First Token task.

    Args:
        cfg: Configuration object with all parameters
    """
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Instantiate model and data module
    ckpt_file = {
        'base': 'tinystories_base__cs1bklll.ckpt',
        'bert-like': 'tinystories_bertlike_embeddings_grad_norm__sqipem6p.ckpt',
        'inv-first': 'tinystories_inv_first_norm__9ecoqzxt.ckpt',
        'identity-grad': 'tinystories_identity_grad_norm__qp6q1mop.ckpt',
    }[cfg.training.method]
    lightning_model, data_module, model_name, cfg = load_model_from_checkpoint(
        cfg.model.output_dir / ckpt_file,
        cfg,
    )

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.training.device is not None:
        torch_device = f'cuda:{cfg.training.device[0]}'
    lightning_model.to(torch_device).eval()

    # Run GCG
    gcg = gcg_algorithm.GCG(
        model=lightning_model.model,
        tokenizer=lightning_model.tokenizer,
        num_prefix_tokens=15,
        num_steps=10_000,
        search_width=1000,
        top_k=64,
    )
    # run_gcg_single_attack(gcg, target_response=' and it was a sunny day.')

    gcg_output_file = cfg.model.output_dir / f'gcg_{model_name}.json'
    if gcg_output_file.exists():
        print(f"File {gcg_output_file} already exists. Skipping GCG evaluation.")
    else:
        run_full_gcg_evaluation(gcg, data_module.val_dataset, gcg_output_file)
    analyze_gcg_results(lightning_model, gcg_output_file)


if __name__ == '__main__':
    main()
