import torch.nn as nn

from src.core.config import ModelConfig
from src.models.factories.resnet18 import ResNet18Factory


class ResNet18Builder:
    """
    Builder for standard (non-spiking) ResNet-18.
    """
    name: str = "resnet18"

    def build(self, cfg: ModelConfig) -> nn.Module:
        if cfg.name.lower() != self.name:
            raise ValueError(
                f"ResNet18Builder cannot build model '{cfg.name}'. Expected '{self.name}'."
            )
        return ResNet18Factory.create(cfg)
