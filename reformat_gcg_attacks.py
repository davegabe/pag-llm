import json
import pathlib
import re
import torch
from data.data_processor import clean_text
from instantiate import load_model_from_checkpoint
from config import CustomLLMPagConfig, apply_config
from gcg.gcg_evaluation import GCGResult
from eval_gcg import _compute_loss_metrics, get_batch_perplexity_from_model
from models.loader import load_model_and_tokenizer

def reformat_successful_attack_strings(json_file_path: pathlib.Path, output_file_path: pathlib.Path, lightning_model, ext_model=None, ext_tokenizer=None, top_k=None):
    """
    Load the GCG results JSON file, compute losses to determine successful attacks,
    reformat the x_attack_str fields for successful ones by replacing \u2581 with spaces, normalizing whitespace,
    and write all reformatted strings to a JSON file with original, attack, and target texts.

    Args:
        json_file_path: Path to the JSON file.
        output_file_path: Path to the output JSON file.
        lightning_model: The loaded Lightning model for loss computation.
        ext_model: External model for perplexity computation (optional).
        ext_tokenizer: External tokenizer for perplexity computation (optional).
        top_k: If specified, only output the top-k results with lowest perplexity.
    """
    print(f"Processing {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter entries with x_attack_str
    results = [GCGResult.from_dict(item) for item in data if "x_attack_str" in item]
    if not results:
        print("No results with x_attack_str")
        return

    # Compute losses to determine success
    orig_losses, attack_losses, kl_divs = _compute_loss_metrics(results, lightning_model)
    if orig_losses is None or attack_losses is None:
        print("Could not compute losses")
        return
    success_mask = attack_losses <= orig_losses

    # Filter successful results
    successful_results = [r for r, success in zip(results, success_mask.tolist()) if success]
    
    if not successful_results:
        print("No successful attacks found")
        return

    # If top_k is specified and we have external model, compute perplexities and sort
    if top_k is not None and ext_model is not None and ext_tokenizer is not None:
        attack_texts = [clean_text(r.x_attack_str, ext_tokenizer) for r in successful_results]

        # Tokenize with external tokenizer
        enc_attack = ext_tokenizer(attack_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        attack_ids = enc_attack['input_ids']
        attack_mask = enc_attack['attention_mask']
        
        # Compute perplexities
        batch_size = 64
        attack_ppls = []
        for i in range(0, len(attack_ids), batch_size):
            end_i = min(i + batch_size, len(attack_ids))
            batch_attack = attack_ids[i:end_i]
            batch_attack_mask = attack_mask[i:end_i]
            attack_ppl = get_batch_perplexity_from_model(ext_model, batch_attack, batch_attack_mask)
            attack_ppls.append(attack_ppl)
        attack_ppls = torch.cat(attack_ppls)
        
        # Sort by perplexity (lowest first) and take top_k
        sorted_indices = torch.argsort(attack_ppls)[:top_k]
        print(f"Selected top {len(sorted_indices)} results with lowest perplexity")

    # Create formatted output
    formatted_results = []
    for x in sorted_indices:
        r = successful_results[x]

        # Decode original prefix and target response using the original model's tokenizer
        original_text = lightning_model.tokenizer.decode(r.original_prefix_ids, skip_special_tokens=True)
        target_text = lightning_model.tokenizer.decode(r.target_response_ids, skip_special_tokens=True)
        
        # Format texts
        attack_text = clean_text(r.x_attack_str, lightning_model.tokenizer)
        original_text = clean_text(original_text, lightning_model.tokenizer)
        target_text = clean_text(target_text, lightning_model.tokenizer)

        formatted_results.append({
            "steps": r.steps,
            "original_text": original_text,
            "attack_text": attack_text,
            "target_text": target_text,
            "perplexity": float(attack_ppls[x]) if top_k is not None and ext_model is not None else None
        })

    # Write to output JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)

    print(f"Reformatted {len(formatted_results)} successful attack strings and saved to {output_file_path}")

@apply_config('pag-identity-small-offline')
def main(cfg: CustomLLMPagConfig):
    top_k_default = 10
    
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
    print(f"Lightning model device: {lightning_model.device}")

    # Load external model for perplexity computation if top_k is specified
    ext_model = None
    ext_tokenizer = None
    if top_k_default is not None:
        # Use external model path from config if available, otherwise use default
        external_model_path = cfg.model.external_llm
        print(f"Loading external model {external_model_path} for perplexity computation")
        try:
            ext_model, ext_tokenizer = load_model_and_tokenizer(str(external_model_path), False)
            ext_model.to(lightning_model.device)  # type: ignore
            ext_model.eval()
            print(f"External model loaded and moved to device: {lightning_model.device}")
        except Exception as e:
            print(f"Failed to load external model: {e}")
            print("Proceeding without top-k filtering")
            ext_model = None

    json_dir = pathlib.Path("/home/dave/Projects/pag-llm/checkpoints/tinystories-pretokenized-base")
    results_path = json_dir / f'gcg_{model_name}.json'

    suffix = f"_top{top_k_default}" if top_k_default is not None else ""
    output_file = pathlib.Path(f"/home/dave/Projects/pag-llm/logs/reformatted_attacks_{model_name}{suffix}.json")

    reformat_successful_attack_strings(results_path, output_file, lightning_model, ext_model, ext_tokenizer, top_k_default)

if __name__ == "__main__":
    main()
