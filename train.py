import os
import logging
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

import config
from data_processor import load_and_process_dataset
from utils import compute_perplexity, save_model_checkpoint, get_optimizer_and_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments:
        --dataset: Name of the dataset to use for training
        --dataset_config: Configuration for the dataset
        --model_checkpoint: Pre-trained model checkpoint to start from
        --output_dir: Directory to save model checkpoints
        --learning_rate: Learning rate for training
        --batch_size: Batch size for training
        --epochs: Number of epochs to train for
        --grad_accum: Gradient accumulation steps
        --max_length: Maximum sequence length
        --text_column: Column name containing the text data
    """
    parser = argparse.ArgumentParser(description="Train LLM model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.DATASET_NAME,
        help="Dataset to use for training"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=config.DATASET_CONFIG,
        help="Dataset configuration"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=config.MODEL_CHECKPOINT,
        help="Model checkpoint to start from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=config.GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=config.MAX_SEQ_LENGTH,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name containing the text data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize accelerator for distributed training
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    device = accelerator.device

    # Log process info
    logger.info(f"Process rank: {accelerator.process_index}, device: {device}")

    # Load tokenizer and model
    logger.info(f"Loading model from {args.model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # Check if tokenizer has padding token, if not set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)

    # Load and process dataset
    logger.info(f"Loading dataset: {args.dataset}")
    train_dataset, eval_dataset = load_and_process_dataset(
        args.dataset, args.dataset_config, tokenizer,
        max_length=args.max_length, text_column=args.text_column
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size
    )

    # Setup optimizer and learning rate scheduler
    num_training_steps = len(train_dataloader) * args.epochs
    optimizer, lr_scheduler = get_optimizer_and_scheduler(
        model, args.learning_rate, config.WEIGHT_DECAY,
        num_training_steps, config.WARMUP_STEPS
    )

    # Prepare for distributed training
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Training loop
    logger.info("Starting training...")
    global_step = 0

    for epoch in range(args.epochs):
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
            if (step + 1) % config.LOGGING_STEPS == 0:
                avg_loss = total_loss / config.LOGGING_STEPS
                logger.info(f"Step {global_step}: Average loss = {avg_loss:.4f}")
                total_loss = 0

            global_step += 1

            # Evaluate model every evaluation steps
            if global_step % config.EVALUATION_STEPS == 0:
                logger.info(f"Evaluating at step {global_step}")
                perplexity = compute_perplexity(model, eval_dataloader, device)
                logger.info(f"Step {global_step}: Perplexity = {perplexity:.4f}")
                model.train()

            # Save checkpoint every save steps
            if global_step % config.SAVE_STEPS == 0:
                logger.info(f"Saving model at step {global_step}")
                unwrapped_model = accelerator.unwrap_model(model)
                save_model_checkpoint(
                    unwrapped_model,
                    tokenizer,
                    args.output_dir,
                    global_step
                )

        # Save model at the end of each epoch
        logger.info(f"Saving model after epoch {epoch+1}")
        unwrapped_model = accelerator.unwrap_model(model)
        save_model_checkpoint(
            unwrapped_model,
            tokenizer,
            args.output_dir,
            f"epoch-{epoch+1}"
        )

    # Save final model
    logger.info("Saving final model")
    unwrapped_model = accelerator.unwrap_model(model)
    save_model_checkpoint(unwrapped_model, tokenizer, args.output_dir)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
