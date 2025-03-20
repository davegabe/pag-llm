import logging

import hydra
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from torch.utils.data import DataLoader
from accelerate import Accelerator

from config import Config
from data_processor import load_and_process_dataset
from utils import compute_perplexity, save_model_checkpoint, get_optimizer_and_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./config", config_name="base")
def main(cfg: Config):
    # Create output directory
    cfg.model.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize accelerator for distributed training
    accelerator = Accelerator(gradient_accumulation_steps=cfg.training.gradient_accumulation_steps)
    device = accelerator.device

    # Log process info
    logger.info(f"Process rank: {accelerator.process_index}, device: {device}")

    # Load tokenizer and model
    logger.info(f"Loading model from {cfg.model.pretrained_base}")
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(cfg.model.pretrained_base)

    # Check if tokenizer has padding token, if not set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(cfg.model.pretrained_base)

    # Load and process dataset
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    train_dataset, eval_dataset = load_and_process_dataset(cfg.dataset, tokenizer, cfg.training.max_seq_length)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=cfg.training.batch_size
    )

    # Setup optimizer and learning rate scheduler
    num_training_steps = len(train_dataloader) * cfg.training.num_epochs
    optimizer, lr_scheduler = get_optimizer_and_scheduler(
        model,
        cfg.training,
        num_training_steps,
    )

    # Prepare for distributed training
    # noinspection PyTypeChecker
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Training loop
    logger.info("Starting training...")
    global_step = 0

    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        for step, batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Log average loss every logging steps
            if (step + 1) % cfg.logging.logging_steps == 0:
                avg_loss = total_loss / cfg.logging.logging_steps
                logger.info(f"Step {global_step}: Average loss = {avg_loss:.4f}")
                total_loss = 0

            global_step += 1

            # Evaluate model every evaluation steps
            if global_step % cfg.logging.evaluation_steps == 0:
                logger.info(f"Evaluating at step {global_step}")
                perplexity = compute_perplexity(model, eval_dataloader, device)
                logger.info(f"Step {global_step}: Perplexity = {perplexity:.4f}")
                model.train()

            # Save checkpoint every save steps
            if global_step % cfg.logging.save_steps == 0:
                logger.info(f"Saving model at step {global_step}")
                unwrapped_model = accelerator.unwrap_model(model)
                save_model_checkpoint(
                    unwrapped_model,
                    tokenizer,
                    cfg.model.output_dir,
                    global_step,
                )

        # Save model at the end of each epoch
        logger.info(f"Saving model after epoch {epoch+1}")
        unwrapped_model = accelerator.unwrap_model(model)
        save_model_checkpoint(
            unwrapped_model,
            tokenizer,
            cfg.model.output_dir,
            global_step,
        )

    # Save final model
    logger.info("Saving final model")
    unwrapped_model = accelerator.unwrap_model(model)
    save_model_checkpoint(unwrapped_model, tokenizer, cfg.model.output_dir)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
