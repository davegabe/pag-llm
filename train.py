import logging
import os

import lightning as pl
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import models.loader as loader
from config import Config, apply_config
from data.data_module import LMDataModule
from models.base_model import BaseLMModel
from models.pag_hidden_model import PAGHiddenModel

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@apply_config()
def train(cfg: Config):
    # Create output directory
    os.makedirs(cfg.model.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    pl.seed_everything(444)

    # Set float32 matmul precision
    torch.set_float32_matmul_precision('medium')
    
    # Load tokenizer and model
    model, tokenizer = loader.load_model_and_tokenizer(
        cfg.model.pretrained_base,
        cfg.training.lora
    )
    model.train()
    
    # Create data module
    data_module = LMDataModule(cfg, tokenizer)
    
    # Select the appropriate model based on training method
    if cfg.training.method == "base":
        lightning_model = BaseLMModel(model, tokenizer, cfg)
    elif cfg.training.method == "pag-hidden":
        # Use the HDF5 file with hidden states for the current model and layer
        hdf5_file_path = f"{cfg.model.output_dir}/hidden_states_layer{cfg.model.hidden_layer_index}.hdf5"
        
        if not os.path.exists(hdf5_file_path):
            logger.error(f"HDF5 file not found at {hdf5_file_path}. Please run hidden_state_collector first.")
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")
            
        lightning_model = PAGHiddenModel(model, tokenizer, cfg, hdf5_file_path)
    else:
        raise ValueError(f"Unknown training method: {cfg.training.method}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model.output_dir,
        filename="model-{step}",
        save_top_k=5,
        verbose=True,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Setup logger
    model_name = cfg.model.pretrained_base.split("/")[-1]
    wandb_logger = WandbLogger(
        project="pag-llm",
        name=f"{model_name}-{cfg.training.method}-{cfg.model.hidden_layer_index}",
        tags=[
            model_name,
            cfg.training.method,
            f"layer-{cfg.model.hidden_layer_index}",
            cfg.dataset.name,
        ],
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.logging.logging_steps,
        val_check_interval=cfg.logging.evaluation_steps,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
    )
    
    # Evaluate model before training
    # trainer.validate(lightning_model, datamodule=data_module)

    # Train model
    trainer.fit(lightning_model, data_module)
    
    # Save final model
    lightning_model.model.save_pretrained(os.path.join(cfg.model.output_dir, "final"))
    
    logger.info("Training completed!")

if __name__ == "__main__":
    train()
