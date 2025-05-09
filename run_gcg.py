import torch

from config import CustomLLMPagConfig, apply_config
from gcg import gcg_algorithm
from instantiate import load_model_from_checkpoint


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
    ckpt_file = 'tinystories_identity_grad_norm__qp6q1mop.ckpt'
    lightning_model, data_module, model_name, cfg = load_model_from_checkpoint(
        cfg.model.output_dir / ckpt_file,
        cfg,
    )
    lightning_model.to('cuda:0')

    # Run GCG
    gcg = gcg_algorithm.GCG(
        model=lightning_model.model,
        tokenizer=lightning_model.tokenizer,
        num_prefix_tokens=15,
        num_steps=10_000,
        search_width=1000,
        top_k=64,
    )
    target_response = ' and it was a sunny day.'
    x_attack_str, y_attack_response, _ = gcg.run(target_response,
                                              evaluate_every_n_steps=50,
                                              stop_after_same_loss_steps=10)
    print(f"Attack string: {x_attack_str}")
    print(f"Attack response: {y_attack_response}")
    print(f"Desired response: {target_response}")


if __name__ == '__main__':
    main()
