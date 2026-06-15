import random
import numpy as np
import torch


def set_random_seed(seed):
    """
    Set the random number seed for Python, NumPy, and PyTorch.

    Parameters:
    seed (int): The seed value to be set for all random number generators.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch's random number generator
    torch.manual_seed(seed)

    # Additionally for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def random_split_dataset(*args, set_type='train', seed=None):
    """
    Splits the datasets into train, validation, and test sets with proportions 8:1:1,
    and returns the specified set (train, val, or test) for each dataset.

    Parameters:
    *args (array-like): Variable number of array-like datasets (e.g., image_paths, sensitives, labels).
    set_type (str): The dataset to return. One of 'train', 'val', or 'test'.

    Returns:
    Specified split of each dataset in the order they were passed.
    """
    # Set the random seed
    if seed is not None:
        np.random.seed(seed)

    # Total number of examples in the first dataset
    total_examples = len(args[0])

    # Creating indices and shuffling them
    indices = np.arange(total_examples)
    np.random.shuffle(indices)

    # Calculate split sizes
    train_end = int(total_examples * 0.6)
    val_end = train_end + int(total_examples * 0.2)

    # Split the indices
    if set_type == 'train':
        selected_indices = indices[:train_end]
    elif set_type == 'val':
        selected_indices = indices[train_end:val_end]
    elif set_type == 'test':
        selected_indices = indices[val_end:]
    elif set_type == 'all':
        selected_indices = indices
    else:
        raise ValueError("Invalid value for set_type. Choose 'train', 'val', or 'test'.")

    # Split the data and return
    return [dataset[selected_indices] for dataset in args]
