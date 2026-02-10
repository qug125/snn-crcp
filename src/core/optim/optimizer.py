# -*- coding: utf-8 -*-
"""
@file: optimizer.py
@description:
    Short description of what this script does.
"""

from src.core.optim.strategies import OPTIMIZER_STRATEGIES
from src.core.config import TrainConfig
import torch

def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    key = cfg.optimizer.lower()
    if key not in OPTIMIZER_STRATEGIES:
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. Options: {list(OPTIMIZER_STRATEGIES)}")
    return OPTIMIZER_STRATEGIES[key].build(model, cfg)
