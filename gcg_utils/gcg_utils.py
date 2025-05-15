import os

import numpy as np
import torch
import transformers


def set_seeds(seed=444):
    """
    Set the random seed for reproducibility (NumPy, PyTorch, and Transformers).
    """
    # Set the random seed for NumPy
    np.random.seed(seed)
    # Use deterministic cuDNN algorithms
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    # If you are using CUDA (i.e., a GPU), also set the seed for it
    torch.cuda.manual_seed_all(seed)
    # Set the hf transformer seed
    transformers.set_seed(seed)
