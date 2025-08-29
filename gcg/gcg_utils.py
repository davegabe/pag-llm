import os

import numpy as np
import torch
import transformers

from config import CustomLLMPagConfig, LLMPagConfig


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

def get_gpu_count(cfg: CustomLLMPagConfig | LLMPagConfig) -> int:
    if cfg.training.gpu_rank is not None:
        # This means that we are running in Cineca
        print('Running with gpu_rank set, using 4 GPUs, as we assume to be in Cineca')
        return 4

    # Determine number of GPUs
    world_size = 0
    if torch.cuda.is_available():
        if cfg.training.device:
            # Use devices specified in config
            visible_devices = [str(d) for d in cfg.training.device]
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(visible_devices)
            world_size = len(visible_devices)
        else:
            # Use all available devices
            world_size = torch.cuda.device_count()
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in range(world_size)])
    return world_size