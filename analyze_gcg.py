import json
from pathlib import Path
import re
import sys
import torch

from config import CustomLLMPagConfig, apply_config
from gcg.gcg_evaluation import GCGResult
from instantiate import load_model_from_checkpoint
from models.base_model import BaseLMModel

TARGET_Y_LEN = 10
ORIGINAL_X_LEN = 20
TOP_K_LOSS = 5

@torch.no_grad()
def get_loss_for_batch(model: BaseLMModel, x_ids: torch.Tensor, x_attention_mask: torch.Tensor, y_ids: torch.Tensor) -> torch.Tensor:
    # Assert that we are padding on the right.
    # Padding on the left breaks the PPL computation.
    assert (x_attention_mask[:, -1] == 1).all(), \
        'Left-side padding is required when computing batch loss for target Y'

    # Infer model device from first parameter
    model_device = next(model.parameters()).device

    if x_ids.device != model_device:
        x_ids = x_ids.to(model_device)
        x_attention_mask = x_attention_mask.to(model_device)
        y_ids = y_ids.to(model_device)

    y_attention_mask = torch.ones_like(y_ids)

    model_input = torch.cat([x_ids, y_ids], dim=1)
    model_attn_mask = torch.cat([x_attention_mask, y_attention_mask], dim=1)

    out = model.model(input_ids=model_input, attention_mask=model_attn_mask, return_dict=True, labels=model_input)
    out_logits: torch.Tensor = out.logits

    # Get only the logits that refer to y_ids, which are the very last tokens, shifted by one
    y_logits = out_logits[:, -y_ids.shape[1]-1:-1, :]
    labels = y_ids

    # shift_logits = y_logits[..., :-1, :].contiguous()
    # shift_labels = y_ids[..., 1:].contiguous()
    # shift_attention_mask_batch = x_attention_mask[..., 1:].contiguous()

    shift_logits = y_logits.contiguous()
    shift_labels = labels.contiguous()
    # print('shift_logits:', shift_logits.shape)
    # print('shift_labels:', shift_labels.shape)

    # # See that the argmax of shift_logits matches the shift_labels
    # print((shift_logits.argmax(dim=-1) == shift_labels).float().mean())

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.transpose(1, 2), shift_labels).sum(1)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return loss


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    if cfg.model.checkpoint_path is None:
        raise ValueError("Model checkpoint path is not set.")
    lightning_model, _, model_name, cfg = load_model_from_checkpoint(
        cfg.model.checkpoint_path, cfg
    )
    # Force to GPU if available
    if torch.cuda.is_available() and lightning_model.device == torch.device('cpu'):
        lightning_model = lightning_model.to('cuda')
        print("Forced lightning model to GPU")
    lightning_model.eval()

    print(f'\n\n=== {model_name.upper()} ===\n')

    # Locate GCG results file based on the loaded config / model name.
    out_dir = Path('gcg_results')
    results_path = out_dir / f'gcg_{model_name}.json'

    with results_path.open('r') as f:
        print(f"Loading GCG results from {results_path}")
        results = [GCGResult.from_dict(r) for r in json.load(f)]

    BATCH_SIZE = 1024
    all_losses = []

    for i in range(0, len(results), BATCH_SIZE):
        print(f'\r{i}/{len(results)}', end='', flush=True, file=sys.stderr)
        batch: list[GCGResult] = results[i:i+BATCH_SIZE]

        y_targets = torch.tensor([
            r.target_response_ids
            for r in batch
        ])
        assert y_targets.shape == (len(batch), TARGET_Y_LEN), \
            f"Expected target shape {(len(batch), TARGET_Y_LEN)}, got {y_targets.shape}"

        x_attack_strings = [
            clean(r.x_attack_str)
            for r in batch
        ]
        x_attack_tokenizer_output = lightning_model.tokenizer(
            x_attack_strings,
            return_tensors='pt',
            padding='longest',
            padding_side='left',    # So that we end up with: [PAD..., X_ATTACK, Y_TARGET]
        )
        x_attack_ids, x_attack_attention_mask = (
            x_attack_tokenizer_output['input_ids'],
            x_attack_tokenizer_output['attention_mask'],
        )
        assert x_attack_ids.shape[0] == len(batch), \
            f"Expected batch size {len(batch)}, got {x_attack_ids.shape[0]}"
        
        x_original_ids = torch.tensor([
            r.original_prefix_ids
            for r in batch
        ])
        assert x_original_ids.shape == (len(batch), ORIGINAL_X_LEN), \
            f"Expected original prefix shape {(len(batch), ORIGINAL_X_LEN)}, got {x_original_ids.shape}"
        x_original_attention_mask = torch.ones_like(x_original_ids)

        attack_losses = get_loss_for_batch(lightning_model, x_attack_ids, x_attack_attention_mask, y_targets)

        original_losses = get_loss_for_batch(lightning_model, x_original_ids, x_original_attention_mask, y_targets)

        # Set the attack loss to infinity if the attack_loss is higher than the original loss
        attack_losses[attack_losses >= original_losses] = float('inf')

        losses = torch.stack([original_losses, attack_losses], dim=1)
        assert losses.shape == (len(batch), 2), \
            f"Expected losses shape {(len(batch), 2)}, got {losses.shape}"

        all_losses.extend(losses.cpu().tolist())
    print('', file=sys.stderr, flush=True)

    # Tell me the top-5 losses (lower is better)
    all_losses = list(enumerate(all_losses))
    all_losses.sort(key=lambda x: x[1][1])  # Sort by attack loss
    print(f"Top {TOP_K_LOSS} losses (lower is better):", all_losses[:TOP_K_LOSS])

    for idx, (original_loss, attack_loss) in all_losses[:TOP_K_LOSS]:
        print('%')

        result = results[idx]
        
        x_original_str = lightning_model.tokenizer.decode(result.original_prefix_ids, skip_special_tokens=True)
        x_original_str = clean(x_original_str)

        x_attack_str = clean(result.x_attack_str)
        
        y_str = lightning_model.tokenizer.decode(result.target_response_ids, skip_special_tokens=True)
        y_str = clean(y_str)

        print('\\midrule')
        print('$\\bx$~: &', x_original_str, '& \\multirow{3}{*}{',y_str,'}  &',f'{original_loss:.2f}','\\\\')
        print('$\\bxa$: &',x_attack_str,'& &  \\textbf{',f'{attack_loss:.2f}','} \\\\')


def clean(txt: str) -> str:
    return re.sub(r'\s+', '', txt).replace("\u2581", " ").strip()


if __name__ == '__main__':
    main() # type: ignore
