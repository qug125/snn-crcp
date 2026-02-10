# -*- coding: utf-8 -*-
"""
@file: cnn.py
@description:
    Short description of what this script does.
"""

import torch
import torch.nn as nn
from src.models.strategies.base import ModelStrategy
class CNNModelStrategy(ModelStrategy):
    """
    Wraps a standard ANN model. reset_state() is a no-op.
    """
    def __init__(self, model: nn.Module):
        self._model = model

    def model(self) -> nn.Module:
        return self._model

    def reset_state(self) -> None:
        return
