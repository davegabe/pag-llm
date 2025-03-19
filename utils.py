import os
import torch
from transformers import get_scheduler
from transformers.optimization import LayerWiseDummyScheduler


def compute_perplexity(model: torch.nn.Module, eval_dataloader: torch.utils.data.DataLoader, device: str) -> float:
    """
    Compute perplexity of the model on the evaluation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        eval_dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (str): Device to run the evaluation on.

    Returns:
        float: The perplexity of the model on the evaluation dataset.
    """
    # Set model to evaluation mode
    model.eval()
    total_loss = 0
    total_tokens = 0

    # Evaluate model
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            # Count non-padding tokens
            non_padding = batch["labels"] != -100
            num_tokens = non_padding.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def save_model_checkpoint(model: torch.nn.Module, tokenizer: LayerWiseDummyScheduler, output_dir: str, step: int = None):
    """
    Save model and tokenizer to the output directory.

    Args:
        model (torch.nn.Module): The model to save.
        tokenizer (LayerWiseDummyScheduler): The tokenizer to save.
        output_dir (str): The output directory.
        step (int): The training step number.
    """
    # Create output directory
    if step:
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    else:
        save_dir = output_dir

    # Save model and tokenizer
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    num_training_steps: int,
    warmup_steps: int,
) -> tuple[torch.optim.Optimizer, LayerWiseDummyScheduler]:
    """
    Prepare optimizer and learning rate scheduler.

    Args:
        model (torch.nn.Module): The model to train.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        num_training_steps (int): The total number of training steps.
        warmup_steps (int): The number of warmup steps.

    Returns:
        tuple[torch.optim.Optimizer, transformers.get_scheduler]: The optimizer and scheduler.
    """
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate
    )

    # Prepare learning rate scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler
