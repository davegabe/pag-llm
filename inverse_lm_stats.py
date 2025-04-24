import pathlib

import torch
from lightning import Trainer

from config import CustomLLMPagConfig, apply_config
from instantiate import load_model_from_checkpoint


@apply_config('inv-first-tiny-train')
def main(cfg: CustomLLMPagConfig):
    torch.set_float32_matmul_precision('medium')

    lightning_module, data_module, module_name, cfg = load_model_from_checkpoint(
        pathlib.Path('./checkpoints/tinystories/tinystories_identity_grad_norm__qp6q1mop.ckpt'),
        cfg,
    )

    trainer = Trainer(devices=cfg.training.device if cfg.training.device else "auto")
    trainer.validate(lightning_module, data_module)


if __name__ == '__main__':
    main()
