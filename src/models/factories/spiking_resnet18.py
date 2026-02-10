# -*- coding: utf-8 -*-
"""
@file: spiking_resnet18.py.py
@description:
    Short description of what this script does.
"""

from typing import Optional
import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet as spiking_resnet

from src.core.config import ModelConfig


class SpikingResNet18Factory:
    """Factory: construct a ResNet-18 with configurable head/channels."""

    @staticmethod
    def create(cfg: ModelConfig) -> nn.Module:
        if cfg.name.lower() != "snn_resnet18":
            raise ValueError(f"Unsupported model name: {cfg.name}")

        if_node = neuron.IFNode
        surrogate_function = surrogate.ATan()

        m = spiking_resnet.spiking_resnet18(
            pretrained=cfg.pretrained,
            spiking_neuron=if_node,
            surrogate_function = surrogate_function,
            detach_reset = True
        )

        m.fc = torch.nn.Linear(m.fc.in_features, 2)

        return m


