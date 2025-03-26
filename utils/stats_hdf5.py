"""
Utility to query and analyze hidden states from the precomputed dataset.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import loader

from config import Config, apply_config
from utils.hdf5 import get_hidden_states_by_next_token, get_count_by_next_token


def get_token_distribution(file_path: str) -> dict:
    """
    Get the distribution of next tokens in the dataset.
    
    Args:
        file_path: Path to the HDF5 file
    
    Returns:
        dict: Dictionary mapping token IDs to their frequency counts
    """
    distribution = {}
    
    with h5py.File(file_path, 'r') as f:
        if 'token_indices' in f:
            for token_id_str in f['token_indices']:
                token_id = int(token_id_str)
                count = len(f['token_indices'][token_id_str])
                distribution[token_id] = count
    
    return distribution


def get_most_common_tokens(file_path: str, top_n: int = 10) -> list[tuple[int, int]]:
    """
    Get the most common next tokens in the dataset.
    
    Args:
        file_path: Path to the HDF5 file
        top_n: Number of most common tokens to return
    
    Returns:
        List[Tuple[int, int]]: List of (token_id, count) tuples
    """
    distribution = get_token_distribution(file_path)
    sorted_tokens = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    return sorted_tokens[:top_n]


def get_hidden_states_by_next_tokens(
        file_path: str,
        token_ids: list[int],
        max_samples_per_token: int = None,
        equal_samples: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieve hidden states for multiple next tokens.
    
    Args:
        file_path: Path to the HDF5 file
        token_ids: List of token IDs to retrieve
        max_samples_per_token: Maximum samples to retrieve per token (None for all)
        equal_samples: Whether to ensure equal number of samples per token
    
    Returns:
        Tuple: (hidden_states, next_tokens, original_indices)
    """
    all_hiddens = []
    all_tokens = []
    all_indices = []
    
    if equal_samples and max_samples_per_token is None:
        # Find minimum count to ensure equal samples
        with h5py.File(file_path, 'r') as f:
            counts = []
            for token_id in token_ids:
                token_id_str = str(token_id)
                if token_id_str in f['token_indices']:
                    counts.append(len(f['token_indices'][token_id_str]))
                else:
                    counts.append(0)
            
            if 0 in counts:
                print(f"Warning: Some tokens have no samples. Tokens without samples: "
                      f"{[t for t, c in zip(token_ids, counts) if c == 0]}")
                # Filter out tokens with no samples
                token_ids = [t for t, c in zip(token_ids, counts) if c > 0]
                counts = [c for c in counts if c > 0]
            
            if not counts:  # If all tokens have no samples
                return np.array([]), np.array([]), np.array([])
                
            max_samples_per_token = min(counts)
    
    # Retrieve hidden states for each token
    for token_id in token_ids:
        hiddens, indices = get_hidden_states_by_next_token(
            file_path, token_id, max_samples=max_samples_per_token
        )
        
        if len(indices) > 0:
            all_hiddens.append(hiddens)
            all_tokens.append(np.full(len(indices), token_id))
            all_indices.append(indices)
    
    # Concatenate all data
    if all_hiddens:
        all_hiddens = np.concatenate(all_hiddens, axis=0)
        all_tokens = np.concatenate(all_tokens, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        return all_hiddens, all_tokens, all_indices
    else:
        return np.array([]), np.array([]), np.array([])


def visualize_token_distribution(file_path: str, tokenizer, top_n: int = 20):
    """
    Visualize the distribution of the most common next tokens.
    
    Args:
        file_path: Path to the HDF5 file
        tokenizer: Tokenizer object
        top_n: Number of most common tokens to visualize
    """
    common_tokens = get_most_common_tokens(file_path, top_n)
    
    # Get token IDs and counts
    token_ids = [t[0] for t in common_tokens]
    counts = [t[1] for t in common_tokens]

    # Convert token IDs to strings and explicitly show \n characters
    token_ids = [
        f"'{tokenizer.decode([t]).replace(chr(10), '\\n')}'\n({t})" for t in token_ids
    ]

    # Set font size
    plt.rcParams.update({'font.size': 6})
    
    # Plot the distribution
    plt.figure(figsize=(20, 6))
    plt.bar(range(len(token_ids)), counts)
    plt.xticks(range(len(token_ids)), [str(t) for t in token_ids])
    plt.xlabel('Token ID')
    plt.ylabel('Frequency')
    plt.title('Distribution of Next Tokens')
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.dirname(file_path)
    plt.savefig(os.path.join(output_dir, 'token_distribution.png'))
    plt.close()


@apply_config()
def main(cfg: Config) -> None:
    """
    Main function for querying and analyzing hidden states.
    
    Args:
        cfg: Configuration object
    """
    print(f"Configuration loaded successfully")

    # Load model and tokenizer
    _, tokenizer = loader.load_model_and_tokenizer(
        cfg.model.pretrained_base
    )
    
    # Determine file path from config
    output_dir = cfg.model.output_dir
    hidden_layer_idx = cfg.model.hidden_layer_index
    file_path = os.path.join(output_dir, f"hidden_states_layer{hidden_layer_idx}.hdf5")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    
    # Set options
    visualize = True
    token_id = 260
    max_samples = 100
    top_n = 50
    
    if visualize:
        print("> Visualizing token distribution...")
        visualize_token_distribution(file_path, tokenizer, top_n=top_n)
        print(f"  Distribution saved to token_distribution.png (top {top_n} tokens)")
    
    if token_id is not None:
        print(f"> Retrieving hidden states for token ID {token_id}...")
        count = get_count_by_next_token(file_path, token_id)
        hiddens, indices = get_hidden_states_by_next_token(
            file_path, token_id, max_samples
        )
        print(f"  Retrieved {len(indices)} samples out of {count}")


if __name__ == "__main__":
    main()
