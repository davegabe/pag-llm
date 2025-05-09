import json
import pathlib

import torch

from config import CustomLLMPagConfig, apply_config
from data.data_processor import TextDataset
from gcg import gcg_algorithm, gcg_evaluation
from instantiate import load_model_from_checkpoint


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
                                                         max_samples_to_attack=1_000)
    with gcg_output_file.open('w') as f:
        json.dump([r.to_dict() for r in gcg_results], f, indent=4)
    print(f"Saved GCG results to {gcg_output_file}")


def analyze_gcg_results(gcg_output_file: pathlib.Path):
    with gcg_output_file.open('r') as f:
        gcg_results = [gcg_evaluation.GCGResult.from_dict(r) for r in json.load(f)]

    # Count the number of successfully attacked tokens
    num_successful_tokens = sum(
        1
        for result in gcg_results
        for target_token, attack_response_token in zip(result.target_response_ids, result.y_attack_response_ids)
        if target_token == attack_response_token
    )
    num_total_tokens = sum(
        min(len(result.target_response_ids), len(result.y_attack_response_ids))
        for result in gcg_results
    )
    token_attack_success_rate = num_successful_tokens / num_total_tokens if num_total_tokens > 0 else 0
    print(f'Token attack success rate: {token_attack_success_rate:.2%}')


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

    torch_device = 'cuda'
    if cfg.training.device is not None:
        torch_device = f'cuda:{cfg.training.device[0]}'
    lightning_model.to(torch_device)

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
    run_full_gcg_evaluation(gcg, data_module.val_dataset, gcg_output_file)
    analyze_gcg_results(gcg_output_file)


if __name__ == '__main__':
    main()
