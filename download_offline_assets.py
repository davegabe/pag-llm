"""
Script to pre-download datasets and tokenizers for offline usage on Cineca.
This should be run before deploying to environments without internet access.
"""

import argparse
import pathlib
import sys
from typing import Optional

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import DatasetConfig, get_config


def download_dataset(dataset_name: str, output_dir: pathlib.Path, splits: Optional[list[str]] = None) -> pathlib.Path:
    """
    Download a dataset to local directory.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., 'DaveGabe/TinyStoriesV2_cleaned-voc2048-seq256-overlap25')
        output_dir: Directory to save the dataset
        splits: List of splits to download. If None, downloads all available splits.
    
    Returns:
        Path to the downloaded dataset directory
    """
    dataset_path = output_dir / "datasets" / dataset_name.replace("/", "_")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset '{dataset_name}' to {dataset_path}")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        
        # If specific splits are requested, filter them
        if splits:
            available_splits = list(dataset.keys())
            for split in splits:
                if split not in available_splits:
                    print(f"Warning: Split '{split}' not found in dataset. Available splits: {available_splits}")
        
        # Save the dataset to disk
        dataset.save_to_disk(str(dataset_path))
        print(f"✓ Dataset '{dataset_name}' downloaded successfully")
        
        return dataset_path
        
    except Exception as e:
        print(f"✗ Error downloading dataset '{dataset_name}': {e}")
        raise


def download_tokenizer(tokenizer_name: str, output_dir: pathlib.Path) -> pathlib.Path:
    """
    Download a tokenizer to local directory.
    
    Args:
        tokenizer_name: HuggingFace tokenizer name
        output_dir: Directory to save the tokenizer
    
    Returns:
        Path to the downloaded tokenizer directory
    """
    tokenizer_path = output_dir / "tokenizers" / tokenizer_name.replace("/", "_")
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading tokenizer '{tokenizer_name}' to {tokenizer_path}")
    
    try:
        # Load and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(str(tokenizer_path))
        print(f"✓ Tokenizer '{tokenizer_name}' downloaded successfully")
        
        return tokenizer_path
        
    except Exception as e:
        print(f"✗ Error downloading tokenizer '{tokenizer_name}': {e}")
        raise


def download_sentence_transformer(model_name: str, output_dir: pathlib.Path) -> pathlib.Path:
    """
    Download a SentenceTransformer model to local directory.
    
    Args:
        model_name: SentenceTransformer model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        output_dir: Directory to save the model
    
    Returns:
        Path to the downloaded model directory
    """
    model_path = output_dir / "sentence_transformers" / model_name.split("/")[-1]
    model_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading SentenceTransformer model '{model_name}' to {model_path}")
    
    try:
        # Load and save the SentenceTransformer model
        model = SentenceTransformer(model_name)
        model.save(str(model_path))
        print(f"✓ SentenceTransformer model '{model_name}' downloaded successfully")
        
        return model_path
        
    except Exception as e:
        print(f"✗ Error downloading SentenceTransformer model '{model_name}': {e}")
        raise


def download_external_llm(model_name: str, model_path: pathlib.Path) -> pathlib.Path:
    """
    Download an external LLM model to local directory.

    Args:
        model_name: Name of the external LLM model
        model_path: Full local path to save the model

    Returns:
        Path to the downloaded model directory
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading external LLM '{model_name}' to {model_path}")

    try:
        # Placeholder for actual LLM download logic
        # This should be replaced with the actual download code for the specific LLM
        print(f"✓ External LLM '{model_name}' downloaded successfully (placeholder)")

        return model_path

    except Exception as e:
        print(f"✗ Error downloading external LLM '{model_name}': {e}")
        raise


def download_from_config(config_path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """
    Download datasets and tokenizers specified in a config file.
    
    Args:
        config_path: Path to the config file
        output_dir: Directory to save downloaded assets
    
    Returns:
        Dictionary mapping asset names to their local paths
    """
    config_path = config_path.resolve()
    config_name = config_path.stem
    
    # Initialize Hydra with the config directory
    cfg = get_config(config_name=config_name)
    
    downloaded_assets = {}
    
    # Check if dataset configuration exists
    if hasattr(cfg, 'dataset') and cfg.dataset:
        dataset_config: DatasetConfig = cfg.dataset
        
        # Download pretokenized dataset if specified
        if dataset_config.use_pretokenized and dataset_config.pretokenized_dataset_name:
            splits = [dataset_config.train_split]
            if dataset_config.eval_split:
                splits.append(dataset_config.eval_split)
            if dataset_config.test_split and dataset_config.test_split != dataset_config.eval_split:
                splits.append(dataset_config.test_split)
            
            dataset_path = download_dataset(
                dataset_config.pretokenized_dataset_name,
                output_dir,
                splits
            )
            downloaded_assets['dataset'] = dataset_path
        
        # Download regular dataset if specified
        elif dataset_config.name:
            splits = [dataset_config.train_split]
            if dataset_config.eval_split:
                splits.append(dataset_config.eval_split)
            if dataset_config.test_split and dataset_config.test_split != dataset_config.eval_split:
                splits.append(dataset_config.test_split)
                
            dataset_path = download_dataset(
                dataset_config.name,
                output_dir,
                splits
            )
            downloaded_assets['dataset'] = dataset_path
        
        # Download tokenizer if specified
        if dataset_config.pretrained_tokenizer_name:
            tokenizer_path = download_tokenizer(
                dataset_config.pretrained_tokenizer_name,
                output_dir
            )
            downloaded_assets['tokenizer'] = tokenizer_path

    # Check if external LLM is specified
    if hasattr(cfg, 'model') and cfg.model:
        model_name = cfg.model.external_llm
        local_path = pathlib.Path(cfg.model.local_external_llm_path)
        download_external_llm(model_name, local_path)
        downloaded_assets['external_llm'] = local_path
    
    return downloaded_assets


def main():
    parser = argparse.ArgumentParser(description="Download datasets and tokenizers for offline usage")
    parser.add_argument("--config", type=pathlib.Path, required=True,
                        help="Path to the config file")
    parser.add_argument("--output-dir", type=pathlib.Path, default="./offline_assets",
                        help="Directory to save downloaded assets (default: ./offline_assets)")
    parser.add_argument("--output-config", type=pathlib.Path,
                        help="Path for the generated offline config (default: <config>-offline.yaml)")
    parser.add_argument("--dataset-only", action="store_true",
                        help="Download only the dataset, not the tokenizer")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="Download only the tokenizer, not the dataset")
    parser.add_argument("--sentence-transformer", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model to download (default: sentence-transformers/all-MiniLM-L6-v2)")
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Config file {args.config} does not exist")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading assets from config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Download assets based on config
    downloaded_assets = download_from_config(args.config, args.output_dir)

    # Download SentenceTransformer model
    sentence_transformer_path = download_sentence_transformer(
        args.sentence_transformer,
        args.output_dir
    )
    downloaded_assets['sentence_transformer'] = sentence_transformer_path

    if not downloaded_assets:
        print("No assets were downloaded. Check your config file.")
        return

    print("\nDownload summary:")
    for asset_type, asset_path in downloaded_assets.items():
        print(f"  {asset_type}: {asset_path}")

    print(f"\n✓ All assets downloaded successfully!")


if __name__ == "__main__":
    main()
