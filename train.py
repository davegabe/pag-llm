import dataclasses
import logging
import os

import lightning as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from config import CustomLLMPagConfig, LLMPagConfig, apply_config
from instantiate import instantiate_model_by_config

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@apply_config('inv-first-tiny-train')
def train(cfg: LLMPagConfig | CustomLLMPagConfig):
    print('Using config:', cfg)

    # Sanity check on WANDB environment variables
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        raise Exception("WANDB_API_KEY environment variable is not set")

    # Create output directory
    os.makedirs(cfg.model.output_dir, exist_ok=True)

    # Set seed for reproducibility
    pl.seed_everything(444)

    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model.output_dir / cfg.training.method,
        filename="model-{step}",
        save_top_k=5,
        verbose=True,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Instantiate model and data module
    lightning_model, data_module, model_name = instantiate_model_by_config(cfg)

    # Setup logger
    run_name = f"{model_name}-{cfg.training.method}"
    tags = [model_name, cfg.training.method, cfg.dataset.name]
    if cfg.training.method == "pag-hidden":
        run_name += f"-{cfg.model.hidden_layer_index}-classes-{cfg.training.pag_classes}"
        tags += [f"layer-{cfg.model.hidden_layer_index}", f"pag-classes-{cfg.training.pag_classes}"]

    wandb_logger = WandbLogger(
        entity=os.environ.get("WANDB_ENTITY", "dwawad"),
        project=os.environ.get("WANDB_PROJECT", "pag-llm"),
        name=run_name,
        tags=tags,
        config= dataclasses.asdict(cfg),
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator="auto",
        devices=cfg.training.device if cfg.training.device else "auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor] if not cfg.training.overfit else None,
        log_every_n_steps=cfg.logging.logging_steps,
        val_check_interval=cfg.logging.evaluation_steps,
        check_val_every_n_epoch=1 if not cfg.training.overfit else None,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
    )

    # Evaluate model before training
    # TIP: you can remove this execution by passing '+training.run_evaluation_before_training=False' as cmd argument
    # Example:  python train.py +training.run_evaluation_before_training=False
    #           -> and it will be skipped, without always commenting this code line.
    if cfg.training.run_evaluation_before_training:
        trainer.validate(lightning_model, datamodule=data_module)

    # Train model
    trainer.fit(lightning_model, data_module)

    # Save final model
    lightning_model.model.save_pretrained(os.path.join(cfg.model.output_dir, "final"))

    logger.info("Training completed!")


if __name__ == "__main__":
    train()
