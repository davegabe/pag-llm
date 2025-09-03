import dataclasses
import logging
import os
from pathlib import Path

import lightning as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger

from config import CustomLLMPagConfig, LLMPagConfig, apply_config
from instantiate import load_model_from_checkpoint

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@apply_config('inv-first-tiny-train')
def test_model(cfg: LLMPagConfig | CustomLLMPagConfig):
    """
    Test a trained model using the same validation approach as training.
    This includes both standard forward evaluation and inverse validation.
    """
    print('Using config for testing:', cfg)

    # Sanity check on WANDB environment variables
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_api_key:
        raise Exception("WANDB_API_KEY environment variable is not set")

    # Set seed for reproducibility
    pl.seed_everything(444)

    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')

    # Load the trained model checkpoint if specified
    checkpoint_path = None
    if cfg.model.checkpoint_path:
        checkpoint_path = cfg.model.checkpoint_path
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        # Use the dedicated function for loading from checkpoint
        lightning_model, data_module, model_name, _ = load_model_from_checkpoint(
            checkpoint_path, cfg
        )

    # Prepare n-gram statistics if the model supports it
    if hasattr(lightning_model, 'prepare_ngram_statistics'):
        lightning_model.prepare_ngram_statistics(data_module)

    # Setup logger for testing
    run_name = f"TEST-{model_name}-{cfg.training.method}"
    tags = ["TEST", model_name, cfg.training.method, cfg.dataset.name]
    if cfg.training.method == "pag-hidden":
        run_name += f"-{cfg.model.hidden_layer_index}-classes-{cfg.training.pag_classes}"
        tags += [f"layer-{cfg.model.hidden_layer_index}", f"pag-classes-{cfg.training.pag_classes}"]

    # Add checkpoint info to tags if available
    if checkpoint_path:
        checkpoint_name = Path(checkpoint_path).stem
        tags.append(f"checkpoint-{checkpoint_name}")

    wandb_logger = WandbLogger(
        entity='pag-llm-team',
        project='pag-llm-test',  # Separate project for test runs
        name=run_name,
        tags=tags,
        config=dataclasses.asdict(cfg),
    )

    # Create trainer for testing
    trainer = pl.Trainer(
        accelerator="auto",
        devices=cfg.training.device if cfg.training.device else "auto",
        logger=wandb_logger,
        log_every_n_steps=cfg.logging.logging_steps,
    )

    # Test the model
    logger.info("Starting model testing...")
    test_results = trainer.test(lightning_model, datamodule=data_module)

    # Log test completion
    logger.info("Testing completed!")
    logger.info(f"Test results: {test_results}")

    return test_results


if __name__ == "__main__":
    test_model()