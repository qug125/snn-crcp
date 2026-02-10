# -*- coding: utf-8 -*-
"""
@file: utils.py
@description:
    Short description of what this script does.
"""

import torch.nn as nn
from src.models.strategies.base import ModelStrategy
from src.models.strategies.cnn import CNNModelStrategy
from src.models.strategies.spikingjelly import SpikingJellyModelStrategy


def wrap_model_as_strategy(model_name: str, model: nn.Module, *, timesteps: int = 4):
    key = model_name.lower()

    if key.startswith("snn") or key.startswith("spiking") or "spiking" in key:
        return SpikingJellyModelStrategy(model=model, timesteps=timesteps)

    return CNNModelStrategy(model=model)
