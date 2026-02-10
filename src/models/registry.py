# -*- coding: utf-8 -*-
"""
@file: registry.py.py
@description:
    Short description of what this script does.
"""

from __future__ import annotations
from typing import Dict
import torch.nn as nn

from src.core.config import ModelConfig
from src.models.builders.base import ModelBuilder
from src.models.builders.resnet18 import ResNet18Builder
from src.models.builders.spiking_resnet18 import SpikingResNet18Builder


MODEL_BUILDERS: Dict[str, ModelBuilder] = {
    ResNet18Builder.name: ResNet18Builder(),
    SpikingResNet18Builder.name: SpikingResNet18Builder(),
}


def build_model(cfg: ModelConfig) -> nn.Module:
    key = cfg.name.lower()
    if key not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{cfg.name}'. Available: {list(MODEL_BUILDERS.keys())}"
        )
    return MODEL_BUILDERS[key].build(cfg)
