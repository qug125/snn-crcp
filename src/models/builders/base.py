# -*- coding: utf-8 -*-
"""
@file: base.py
@description:
    Short description of what this script does.
"""

from typing import Protocol
import torch.nn as nn
from src.core.config import ModelConfig


class ModelBuilder(Protocol):
    """
    Builder interface for constructing nn.Module from ModelConfig.
    Used by the model registry and CLI.
    """
    name: str

    def build(self, cfg: ModelConfig) -> nn.Module:
        ...
