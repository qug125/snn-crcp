from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, sets PyTorch/CUDA deterministic flags.
            Note: Deterministic settings can reduce throughput.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
