"""Miscellaneous utilities."""

import random

import numpy as np
import torch


def seed_all(seed: int | np.uint32) -> None:
    """Sets a random seed for all relevant libraries.

    Args:
        seed: A random seed.
    """
    random.seed(a=int(seed))
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=int(seed))
    torch.cuda.manual_seed_all(seed=int(seed))
