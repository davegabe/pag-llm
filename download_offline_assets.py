#!/usr/bin/env python3
"""
Script to pre-download datasets and tokenizers for offline usage on Cineca.
This should be run before deploying to environments without internet access.
"""

import argparse
import pathlib
import sys
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from config import DatasetConfig


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
    config_dir = config_path.parent
    config_name = config_path.stem
    
    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    
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
    
    return downloaded_assets


def generate_offline_config(config_path: pathlib.Path, downloaded_assets: dict[str, pathlib.Path], 
                          output_config_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Generate a modified config file with local paths for offline usage.
    
    Args:
        config_path: Original config file path
        downloaded_assets: Dictionary of downloaded asset paths
        output_config_path: Where to save the offline config. If None, uses original name with '-offline' suffix.
    
    Returns:
        Path to the generated offline config
    """
    config_path = config_path.resolve()
    
    if output_config_path is None:
        output_config_path = config_path.parent / f"{config_path.stem}-offline{config_path.suffix}"
    
    # Read the original config
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Replace remote paths with local paths
    if 'dataset' in downloaded_assets:
        dataset_path = downloaded_assets['dataset']
        # Add local_dataset_path and comment out remote dataset name
        config_content = config_content.replace(
            f"  use_pretokenized: True",
            f"  use_pretokenized: True\n  local_dataset_path: '{dataset_path}'"
        )
        
        # Comment out the remote dataset name to force local loading
        if "pretokenized_dataset_name:" in config_content:
            config_content = config_content.replace(
                "  pretokenized_dataset_name:",
                "  # pretokenized_dataset_name:"
            )
    
    if 'tokenizer' in downloaded_assets:
        tokenizer_path = downloaded_assets['tokenizer']
        # Add local_tokenizer_path
        if "local_dataset_path:" in config_content:
            config_content = config_content.replace(
                f"  local_dataset_path: '{downloaded_assets['dataset']}'",
                f"  local_dataset_path: '{downloaded_assets['dataset']}'\n  local_tokenizer_path: '{tokenizer_path}'"
            )
        else:
            config_content = config_content.replace(
                f"  use_pretokenized: True",
                f"  use_pretokenized: True\n  local_tokenizer_path: '{tokenizer_path}'"
            )
        
        # Comment out the remote tokenizer name
        if "pretrained_tokenizer_name:" in config_content:
            config_content = config_content.replace(
                "  pretrained_tokenizer_name:",
                "  # pretrained_tokenizer_name:"
            )
    
    # Write the offline config
    with open(output_config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Offline config generated: {output_config_path}")
    return output_config_path


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
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Config file {args.config} does not exist")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading assets from config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Download assets based on config
        downloaded_assets = download_from_config(args.config, args.output_dir)
        
        if not downloaded_assets:
            print("No assets were downloaded. Check your config file.")
            return
        
        print("\nDownload summary:")
        for asset_type, asset_path in downloaded_assets.items():
            print(f"  {asset_type}: {asset_path}")
        
        # Generate offline config
        offline_config_path = generate_offline_config(
            args.config, 
            downloaded_assets, 
            args.output_config
        )
        
        print(f"\n✓ All assets downloaded successfully!")
        print(f"To use offline assets, use config: {offline_config_path}")
        
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
