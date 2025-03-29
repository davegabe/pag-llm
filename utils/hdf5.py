import gc
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm


def _initialize_hdf5_datasets(
    f: h5py.File,
    prefix_len: int,
    hidden_dim: int,
    chunk_size: int = 64,
    compression_level: int = 4
):
    """
    Initialize HDF5 datasets for hidden states and next tokens.

    Args:
        f: HDF5 file object
        prefix_len: Length of the prefix sequence
        hidden_dim: Dimensionality of the hidden states
        chunk_size: Size of chunks for HDF5 datasets (affects read efficiency)
        compression_level: Level of compression (0-9, where 9 is highest)
    """
    f.attrs['hidden_dim'] = hidden_dim
    f.attrs['prefix_len'] = prefix_len
    f.attrs['total_samples'] = 0

    if 'created_timestamp' not in f.attrs:
        f.attrs['created_timestamp'] = str(np.datetime64('now'))

    # Create resizable datasets with optimized chunking for sample-wise access
    f.create_dataset(
        'hidden_states',
        shape=(0, prefix_len, hidden_dim),
        maxshape=(None, prefix_len, hidden_dim),
        chunks=(chunk_size, prefix_len, hidden_dim),
        dtype='f4',
        compression='gzip',
        compression_opts=compression_level
    )

    f.create_dataset(
        'next_tokens',
        shape=(0, prefix_len),
        maxshape=(None, prefix_len),
        chunks=(chunk_size, prefix_len),
        dtype='i4',
        compression='gzip',
        compression_opts=compression_level
    )

    # Create token indices group for efficient token-based retrieval
    f.create_group('token_indices')


def _update_token_indices(
    f: h5py.File,
    next_tokens_np: np.ndarray,
    current_size: int,
    batch_size: int,
    compression_level: int = 4
):
    """
    Update token index mapping for efficient retrieval.

    Args:
        f: HDF5 file object
        next_tokens_np: Next token IDs as a NumPy array
        current_size: Current number of samples in the file
        batch_size: Number of samples in the current batch
    """
    # We only use the first next token for each sample (index 0)
    for i in range(batch_size):
        # Cast to Python int for use as key
        token_id = int(next_tokens_np[i, 0])
        token_id_str = str(token_id)

        # Create or update index arrays for this token
        if token_id_str in f['token_indices']:
            # Get existing indices, resize and update
            indices_dset = f['token_indices'][token_id_str]
            old_size = indices_dset.shape[0]
            indices_dset.resize((old_size + 1,))
            indices_dset[old_size] = current_size + i
        else:
            # Create new dataset for this token
            f['token_indices'].create_dataset(
                token_id_str,
                data=np.array([current_size + i]),
                maxshape=(None,),
                dtype='i4',
                chunks=(128,),  # Optimize for frequent appends
                compression='gzip',
                compression_opts=compression_level
            )


# noinspection GrazieInspection
def save_hidden_states_to_hdf5(
    hidden_states: torch.Tensor,
    next_tokens: torch.Tensor,
    file_path: str,
    append: bool = False,
    chunk_size: int = 64,
    compression_level: int = 4
) -> int:
    """
    Save hidden states and next token IDs to an HDF5 file.

    Args:
        hidden_states: Tensor of hidden states [batch_size, prefix_len, hidden_dim]
        next_tokens: Tensor of next token IDs [batch_size, prefix_len]
        file_path: Path to save the HDF5 file
        append: Whether to append to existing file or create a new one
        chunk_size: Size of chunks for HDF5 datasets (affects read efficiency)
        compression_level: Level of compression (0-9, where 9 is highest)

    Returns:
        int: Total number of samples in the file after saving
    """
    mode = 'a' if append and os.path.exists(file_path) else 'w'

    # Convert to numpy for h5py compatibility
    hidden_states_np = hidden_states.cpu().numpy()
    next_tokens_np = next_tokens.cpu().numpy()

    try:
        with h5py.File(file_path, mode) as f:
            batch_size, prefix_len, hidden_dim = hidden_states_np.shape

            # Check if the required datasets exist
            datasets_exist = 'hidden_states' in f and 'next_tokens' in f

            # Store dataset info if this is a new file or datasets don't exist
            if mode == 'w' or not datasets_exist:
                _initialize_hdf5_datasets(
                    f, prefix_len, hidden_dim,
                    chunk_size, compression_level
                )

            # Get current size and calculate new size after adding this batch
            current_size = f['hidden_states'].shape[0]
            new_size = current_size + batch_size

            # Resize datasets
            f['hidden_states'].resize(new_size, axis=0)
            f['next_tokens'].resize(new_size, axis=0)

            # Add new data
            f['hidden_states'][current_size:new_size] = hidden_states_np
            f['next_tokens'][current_size:new_size] = next_tokens_np

            # Update token index mapping for efficient retrieval
            _update_token_indices(
                f, next_tokens_np, current_size,
                batch_size, compression_level
            )

            # Update metadata
            f.attrs['total_samples'] = new_size
            f.attrs['last_updated'] = str(np.datetime64('now'))

            return new_size
    except Exception as e:
        print(f"Error occurred while saving to HDF5: {str(e)}")
        raise


def get_hidden_states_by_next_token(
    file_path: str,
    token_id: int,
    max_samples: int = None,
    randomize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Efficiently retrieve hidden states with a specific next token.

    Args:
        file_path: Path to the HDF5 file
        token_id: The token ID to retrieve samples for
        max_samples: Maximum number of samples to retrieve (None for all)
        randomize: Whether to shuffle the samples

    Returns:
        tuple: (hidden_states, indices) where indices are the original positions in the dataset
    """
    try:
        with h5py.File(file_path, 'r') as f:
            token_id_str = str(token_id.item() if isinstance(token_id, torch.Tensor) else token_id)

            if 'token_indices' not in f or token_id_str not in f['token_indices']:
                print(f"No samples found with next token ID {token_id}")
                return np.empty((0, f.attrs['prefix_len'], f.attrs['hidden_dim'])), np.empty(0)

            # Get indices of samples with this token
            indices = f['token_indices'][token_id_str]
            size = len(indices)

            # Get random indices if required
            if randomize:
                indices = np.random.permutation(size)[:max_samples]
            else:
                indices = indices[:max_samples]

            # Get the hidden states for these indices
            hidden_states = f['hidden_states'][indices]

            return hidden_states, indices
    except Exception as e:
        print(f"Error retrieving hidden states for token {token_id}: {str(e)}")
        raise

def get_count_by_next_token(
    file_path: str,
    token_id: int
) -> int:
    """
    Get the number of samples with a specific next token.

    Args:
        file_path: Path to the HDF5 file
        token_id: The token ID to retrieve samples for

    Returns:
        int: Number of samples with the specified next token
    """
    try:
        with h5py.File(file_path, 'r') as f:
            token_id_str = str(token_id)

            if 'token_indices' not in f or token_id_str not in f['token_indices']:
                return 0

            return len(f['token_indices'][token_id_str])
    except Exception as e:
        print(f"Error retrieving count for token {token_id}: {str(e)}")
        raise

def get_all_next_tokens(file_path: str) -> torch.Tensor:
    """
    Get all unique next tokens present in the HDF5 file.

    Args:
        file_path: Path to the HDF5 file
    
    Returns:
        torch.Tensor: Tensor of all unique next tokens
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if 'token_indices' not in f:
                return torch.tensor([])

            tokens = {int(k) for k in f['token_indices'].keys()}
            return torch.tensor(list(tokens))
    except Exception as e:
        print(f"Error retrieving all next tokens: {str(e)}")
        raise

def main():
    # Test the functions
    batch_size = 32
    hidden_dim = 768
    prefix_len = 40
    n_append = 10
    desired_next_token = 2

    # Set seed for reproducibility
    torch.manual_seed(444)

    file_path = 'test_hidden_states.h5'

    # Random tensor for testing
    for _ in tqdm(range(n_append), desc="Appending random tensors to HDF5"):
        # Create random tensors
        hidden_states = torch.randn(batch_size, prefix_len, hidden_dim)
        next_tokens = torch.randint(0, 10, (batch_size, 1))

        # Save hidden states to HDF5
        save_hidden_states_to_hdf5(hidden_states, next_tokens, file_path, append=True)

        del hidden_states, next_tokens
        torch.cuda.empty_cache()
        gc.collect()

    # Retrieve hidden states for desired_next_token
    print(f"Retrieving hidden states for next token ID {desired_next_token}")
    hidden_states_retrieved, indices = get_hidden_states_by_next_token(
        file_path,
        desired_next_token
    )
    print(hidden_states_retrieved.shape, indices.shape)
    print(indices)

    # Clean up
    os.remove(file_path)


if __name__ == "__main__":
    main()
