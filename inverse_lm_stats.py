import pathlib

import torch
from lightning import Trainer

from instantiate import load_model_from_checkpoint


def main():
    lightning_module, data_module, module_name = load_model_from_checkpoint(
        pathlib.Path('./checkpoints/tinystories/tinystories_identity_grad_norm__qp6q1mop.ckpt'),
        torch.device('cpu'),
    )
    print(lightning_module)

    trainer = Trainer()
    trainer.test(lightning_module, data_module)


if __name__ == '__main__':
    main()
