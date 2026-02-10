from __future__ import annotations

import torch.nn as nn

from src.core.config import ModelConfig
from src.models.factories.spiking_resnet18 import SpikingResNet18Factory


class SpikingResNet18Builder:
    """
    Builder for SpikingJelly-based Spiking ResNet-18.
    """
    name: str = "snn_resnet18"

    def build(self, cfg: ModelConfig) -> nn.Module:
        if cfg.name.lower() != self.name:
            raise ValueError(
                f"SpikingResNet18Builder cannot build model '{cfg.name}'. Expected '{self.name}'."
            )
        return SpikingResNet18Factory.create(cfg)
