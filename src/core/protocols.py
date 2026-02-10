# -*- coding: utf-8 -*-
"""
@file: protocols.py.py
@description:
    Short description of what this script does.
"""

import torch
import torch.nn as nn
from typing import Protocol, Dict, Any

from src.core.config import ModelConfig
class ModelBuilder(Protocol):
    name: str
    def build(self, cfg: ModelConfig) -> nn.Module: ...



class LossStrategy(Protocol):
    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

class Callback(Protocol):
    """Observer/Hook style callbacks for extensibility."""
    def on_train_begin(self, state: Dict[str, Any]) -> None: ...
    def on_epoch_end(self, state: Dict[str, Any]) -> None: ...
    def on_train_end(self, state: Dict[str, Any]) -> None: ...
