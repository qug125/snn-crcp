# -*- coding: utf-8 -*-
"""
@file: scheduler.py
@description:
    Short description of what this script does.
"""

# src/core/optim/scheduler.py
from typing import Optional, Tuple
from src.core.optim.strategies import SCHEDULER_STRATEGIES
from src.core.config import TrainConfig
from torch.optim import Optimizer

def build_scheduler(optimizer: Optimizer, cfg: TrainConfig, *, steps_per_epoch: int):
    key = cfg.scheduler.lower()
    if key in ("none", "", None):
        return None
    if key not in SCHEDULER_STRATEGIES:
        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'. Options: {list(SCHEDULER_STRATEGIES)}")
    return SCHEDULER_STRATEGIES[key].build(optimizer, cfg, steps_per_epoch)
