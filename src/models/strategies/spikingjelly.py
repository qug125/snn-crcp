# -*- coding: utf-8 -*-
"""
@file: spikingjelly.py
@description:
    Short description of what this script does.
"""

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, encoding
from src.models.strategies.base import ModelStrategy


class SpikingJellyWrappedModule(nn.Module):
    """
    Wrap a step-based spiking model to accept static images (B,C,H,W) and
    run for T timesteps using rate coding, returning time-aggregated logits.
    """
    def __init__(self, snn: nn.Module, T: int):
        super().__init__()
        self.snn = snn
        self.encoder = encoding.PoissonEncoder()
        self.T = int(T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) in float (already normalized)
        # Rate coding: Bernoulli spikes with probability proportional to pixel intensity.
        # Since normalized images include negatives, map to [0,1] first.
        # Simple, stable mapping: sigmoid -> [0,1]
        # p = torch.sigmoid(x)
        outs = []
        for _ in range(self.T):
            xt = self.encoder(x)
            out = self.snn(xt)
            outs.append(out)

        # Aggregate across time (mean)
        return torch.stack(outs, dim=0).mean(dim=0)


class SpikingJellyModelStrategy:
    def __init__(self, model: nn.Module, timesteps: int):
        self._base = model
        self._wrapped = SpikingJellyWrappedModule(model, timesteps)

    def model(self) -> nn.Module:
        return self._wrapped

    def reset_state(self) -> None:
        functional.reset_net(self._base)
