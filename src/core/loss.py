# -*- coding: utf-8 -*-
"""
@file: loss.py.py
@description:
    Short description of what this script does.
"""

__author__ = "Nick Littlefield"
__email__ = "ngl18@pitt.edu"
__version__ = "0.1.0"

from src.core.protocols import LossStrategy

import torch
import torch.nn.functional as F

class SpikeRateMSELossStrategy(LossStrategy):
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        # Multi-class style: [B, C]
        if outputs.dim() == 2 and outputs.size(1) > 1:
            C = outputs.size(1)
            y = F.one_hot(targets.long().view(-1), num_classes=C).float()
            loss = F.mse_loss(outputs, y)
            return loss

        # Binary single-output: [B] or [B,1]
        out = outputs.view(-1)
        y = targets.float().view(-1)
        loss = F.mse_loss(out, y)
        return loss

class CrossEntropyLossStrategy(LossStrategy):
    """
    Standard CNN-style classification loss.
    Assumes model outputs raw logits of shape [B, C].
    """

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # -------- sanity checks --------
        if logits.dim() != 2:
            raise ValueError(
                f"CrossEntropyLoss expects logits [B, C], got {tuple(logits.shape)}"
            )

        if targets.dim() != 1:
            targets = targets.view(-1)

        targets = targets.long()

        # -------- loss --------
        loss = F.cross_entropy(
            logits,
            targets
        )


        return loss
