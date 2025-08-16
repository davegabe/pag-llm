import dataclasses
import pathlib
from pathlib import Path
from typing import Any

import torch
from evaluate.utils import logging
from lightning.pytorch.loggers import WandbLogger
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CustomLLMPagConfig, apply_config
from evaluation_metrics import BackwardInferenceEvaluator, aggregate_metrics
from instantiate import load_model_from_checkpoint
from utils.backward_inference_result import BackwardInferenceResult


def load_semantic_model(cfg: CustomLLMPagConfig) -> SentenceTransformer:
    """
    Load SentenceTransformer model from local path if available, otherwise from HuggingFace.

    Args:
        cfg: Configuration object containing local model path if available

    Returns:
        SentenceTransformer: Loaded model
    """
    if cfg.dataset.local_sentence_transformer_path is not None:
        print(f"Loading SentenceTransformer model from local path: {cfg.dataset.local_sentence_transformer_path}")
        return SentenceTransformer(cfg.dataset.local_sentence_transformer_path)
    else:
        print("Loading SentenceTransformer model from HuggingFace: sentence-transformers/all-MiniLM-L6-v2")
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def load_bigram_from_file(bigram_file: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load the bigram from the file.
    :param bigram_file: Path to the bigram file.
    :return: bigram dictionary or None if the file does not exist.
    """
    if not bigram_file.exists():
        return None

    loaded_file = torch.load(str(bigram_file.resolve()), map_location='cpu')

    reverse_bigram = loaded_file.get('bigram')
    bigram_counts = loaded_file['distribution_after_token']
    return reverse_bigram, bigram_counts


def build_and_save_bigram(train_dataloader: DataLoader, vocab_size: int, bigram_file: pathlib.Path) -> tuple[
    torch.Tensor, torch.Tensor]:
    distribution_after_token = torch.zeros((vocab_size, vocab_size),
                                           dtype=torch.int)  # [k+1 token] -> [k token] -> count

    for batch in tqdm(train_dataloader, desc='Building bigram'):
        for i_sample in range(batch.input_ids.size(0)):
            sample_len = batch.attention_mask[i_sample].sum().item()
            for k in range(sample_len - 1, 0, -1):
                # Get the k-th token
                k_token_id = batch.input_ids[i_sample, k]

                # Get the k+1-th token
                k_minus_one_token_id = batch.input_ids[i_sample, k - 1]

                # Count the occurrences of the k-th and (k-1)-th tokens
                distribution_after_token[k_token_id, k_minus_one_token_id] += 1

    # Build the bigram
    print('Building bigram...')
    # Find the most frequent token for each (k+1)-th token
    torch_bigram = torch.argmax(distribution_after_token, dim=1)

    # Save the bigram to the file
    bigram_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save({'distribution_after_token': distribution_after_token, 'bigram': torch_bigram, },
               str(bigram_file.resolve()))

    return torch_bigram, distribution_after_token


def init_evaluation(cfg: CustomLLMPagConfig, device: str, use_init: str, ckpt_file: str):
    torch.set_float32_matmul_precision('medium')

    # Decide the strategy for the init
    allowed_init = {'bigram', 'random', 'pad', 'bos'}
    assert use_init in allowed_init, f'Invalid initialization strategy: {use_init}. Allowed values are: {allowed_init}'

    lightning_module, data_module, module_name, cfg = load_model_from_checkpoint(ckpt_file, cfg, )
    lightning_module.to(device)

    model_class_name = lightning_module.__class__.__name__

    print()
    print("TESTING INVERSE LM")
    print(" - Model: ", model_class_name)
    print(" - Init strategy: ", use_init.upper())
    print()

    train_bigram_file = cfg.model.output_dir / f'train_bigram_{cfg.model.vocab_size}_full.pt'
    print('Bigram file:', train_bigram_file)
    if train_bigram_file.exists():
        # Load the bigram from the file
        reverse_bigram, bigram_counts = load_bigram_from_file(train_bigram_file)
    else:
        # Build the bigram from the training data
        reverse_bigram, bigram_counts = build_and_save_bigram(data_module.train_dataloader(), cfg.model.vocab_size,
                                                              train_bigram_file)
    reverse_bigram, bigram_counts = map(lambda x: x.to(device), (reverse_bigram, bigram_counts))

    # To always use special token as the filler for the unknown token:
    if use_init == 'bos':
        reverse_bigram = torch.full_like(reverse_bigram, lightning_module.tokenizer.bos_token_id)
    elif use_init == 'pad':
        reverse_bigram = torch.full_like(reverse_bigram, lightning_module.tokenizer.pad_token_id)

    return lightning_module, model_class_name, data_module, reverse_bigram, bigram_counts


def run_evaluation(device: str, precomputed_inference_json_path: str, cfg: CustomLLMPagConfig):
    # Load the samples
    print('Loading precomputed inference results from:', precomputed_inference_json_path)
    inference_result = BackwardInferenceResult.from_file(precomputed_inference_json_path)

    use_init = inference_result.use_init
    ckpt_file = inference_result.ckpt_file
    prefix_len = inference_result.prefix_len
    baseline_ckpt_file = inference_result.baseline_ckpt_file
    k_samples = inference_result.k_samples
    skip_prefix_tokens = inference_result.skip_prefix_tokens
    beam_size = inference_result.beam_size

    # Setup WandB logger for backward inference evaluation
    run_name = f"backward-{cfg.training.method}-{use_init}"
    tags = ["backward", cfg.training.method, cfg.dataset.name, use_init]
    if cfg.training.method == "pag-hidden":
        run_name += f"-{cfg.model.hidden_layer_index}-classes-{cfg.training.pag_classes}"
        tags += [f"layer-{cfg.model.hidden_layer_index}", f"pag-classes-{cfg.training.pag_classes}"]

    # Add checkpoint info to tags if available
    if ckpt_file:
        checkpoint_name = Path(ckpt_file).stem
        tags.append(f"checkpoint-{checkpoint_name}")

    wandb_logger = WandbLogger(entity='pag-llm-team', project='pag-llm-backward-inference',
                               # Separate project for backward inference runs
                               name=run_name, tags=tags, config={**dataclasses.asdict(cfg),
                                                                 'evaluation_params': {'device': device,
                                                                                       'prefix_len': prefix_len,
                                                                                       'use_init': use_init,
                                                                                       'ckpt_file': ckpt_file,
                                                                                       'baseline_ckpt_file': baseline_ckpt_file,
                                                                                       'k_samples': k_samples,
                                                                                       'skip_prefix_tokens': skip_prefix_tokens,
                                                                                       'beam_size': beam_size, }}, )

    # Initialize semantic similarity model
    print("Loading SentenceTransformer model for semantic similarity...")
    semantic_model = load_semantic_model(cfg)

    lightning_module, _, _, _, _ = init_evaluation(cfg=cfg, device=device, use_init=use_init, ckpt_file=ckpt_file, )

    external_llm = str(cfg.model.local_external_llm_path) or cfg.model.external_llm

    # Initialize the comprehensive evaluator with the forward model
    # For forward coherence, we use the same model (could use a different forward model if available)
    evaluator = BackwardInferenceEvaluator(forward_model=lightning_module, external_llm=external_llm,
                                           semantic_model=semantic_model)

    lightning_module.eval()

    for i, sample in enumerate(tqdm(inference_result.samples, desc='Evaluating samples')):
        # Skip the sample if it is already evaluated
        if sample.ilm_metrics is not None:
            continue

        # Create tensors for the list[int] tokens
        original_prefix_tokens = torch.tensor(sample.original_prefix_tokens, dtype=torch.long, device=device)
        predicted_prefix_tokens = torch.tensor(sample.predicted_prefix_tokens, dtype=torch.long, device=device)
        suffix_tokens = torch.tensor(sample.suffix_tokens, dtype=torch.long, device=device)

        # Get the strings of text
        original_prefix_text = sample.original_prefix_text
        predicted_prefix_text = sample.predicted_prefix_text
        suffix_text = sample.suffix_text

        ilm_generated_metrics = evaluator.compute_comprehensive_metrics(reference_prefix=original_prefix_tokens,
                                                                        generated_prefix=predicted_prefix_tokens,
                                                                        true_suffix=suffix_tokens,
                                                                        reference_prefix_mask=torch.ones_like(
                                                                            original_prefix_tokens),
                                                                        generated_prefix_mask=torch.ones_like(
                                                                            predicted_prefix_tokens),
                                                                        suffix_mask=torch.ones_like(suffix_tokens),
                                                                        predicted_overall_text=predicted_prefix_text,
                                                                        original_overall_text=original_prefix_text,
                                                                        suffix_text=suffix_text, )
        sample.ilm_metrics = ilm_generated_metrics

        if sample.bigram_text:
            bigram_tokens = torch.tensor(sample.bigram_tokens, dtype=torch.long, device=device)
            bigram_text = sample.bigram_text

            bigram_generated_metrics = evaluator.compute_comprehensive_metrics(reference_prefix=original_prefix_tokens,
                                                                               generated_prefix=bigram_tokens,
                                                                               true_suffix=suffix_tokens,
                                                                               reference_prefix_mask=torch.ones_like(
                                                                                   original_prefix_tokens),
                                                                               generated_prefix_mask=torch.ones_like(
                                                                                   bigram_tokens),
                                                                               suffix_mask=torch.ones_like(
                                                                                   suffix_tokens),
                                                                               predicted_overall_text=bigram_text,
                                                                               original_overall_text=original_prefix_text,
                                                                               suffix_text=suffix_text, )
            sample.bigram_metrics = bigram_generated_metrics

        # Every 100 samples, save the samples to disk
        if i % 100 == 0 and i > 0:
            print(f"Saving intermediate results after sample {i}...")
            inference_result.to_file(precomputed_inference_json_path)

    # Initialize metrics tracking (new evaluation metrics)
    all_ilm_generated_metrics: list[dict] = [
        sample.ilm_metrics for sample in inference_result.samples
    ]
    all_bigram_generated_metrics: list[dict] = [
        sample.bigram_metrics for sample in inference_result.samples if sample.bigram_metrics is not None
    ]

    # Aggregate comprehensive metrics
    aggregated_comprehensive_metrics: dict[str, Any] = aggregate_metrics(all_ilm_generated_metrics)

    # Add an "ilm/" prefix to all the keys
    aggregated_comprehensive_metrics = {'ilm/' + k: v for k, v in aggregated_comprehensive_metrics.items()}

    if all_bigram_generated_metrics:
        # Use a different prefix for bigram metrics (e.g., "bigram/")
        aggregated_bigram_metrics = aggregate_metrics(all_bigram_generated_metrics)
        aggregated_comprehensive_metrics.update({'bigram/' + k: v for k, v in aggregated_bigram_metrics.items()})

    wandb_logger.experiment.log(aggregated_comprehensive_metrics)

    # Finish WandB run
    wandb_logger.experiment.finish()


@apply_config('inv-first-tiny-train-small')
def main(cfg: CustomLLMPagConfig):
    logging.disable_progress_bar()

    if "inv-first" in cfg.training.method:
        print(f"Method {cfg.training.method} need to use BOS for initialization, ")
        use_init = 'bos'
    elif "bert-like" in cfg.training.method or "pag-mix-identity-score-embeddings" in cfg.training.method:
        print(f"Method {cfg.training.method} need to use PAD for initialization, ")
        use_init = 'pad'
    elif "identity-grad" in cfg.training.method or cfg.training.method == "base":
        print(f"Method {cfg.training.method} need to use PAG for initialization, ")
        use_init = 'bigram'
    else:
        raise ValueError(f"Unsupported training method: {cfg.training.method}. ")

    output_file = f'backward_inference-{cfg.training.method}-{use_init}.zip'

    run_evaluation(device='cuda:0',
                   precomputed_inference_json_path=f'{cfg.model.output_dir}/{output_file}',
                   cfg=cfg)


if __name__ == '__main__':
    main()
