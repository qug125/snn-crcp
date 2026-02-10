# -*- coding: utf-8 -*-
"""
@file: resnet18.py.py
@description:
    Short description of what this script does.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from typing import List
from src.core.config import ModelConfig

class ResNet18Factory:
    """Factory: construct a ResNet-18 with configurable head/channels."""
    @staticmethod
    def create(cfg: ModelConfig) -> nn.Module:
        if cfg.name.lower() != "resnet18":
            raise ValueError(f"Unsupported model name: {cfg.name}")

        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if cfg.pretrained else None)

        # adjust input channels if needed
        if cfg.input_channels != 3:
            old_conv = m.conv1
            m.conv1 = nn.Conv2d(
                cfg.input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # if pretrained and channels differ, you may want a smarter init. Here: Kaiming.
            nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")

        # replace classifier head
        in_features = m.fc.in_features
        head: List[nn.Module] = []
        if cfg.dropout_p and cfg.dropout_p > 0:
            head.append(nn.Dropout(p=cfg.dropout_p))
        head.append(nn.Linear(in_features, cfg.num_classes))
        m.fc = nn.Sequential(*head)

        return m
