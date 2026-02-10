# -*- coding: utf-8 -*-
"""
@file: metrics.py
@description:
    Short description of what this script does.
"""

from dataclasses import dataclass
import torch
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC


class MetricsStrategy:
    name: str = "metric"

    def to(self, device: torch.device) -> "MetricsStrategy":
        return self

    def reset(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> float:
        raise NotImplementedError


@dataclass
class AUROCStrategy(MetricsStrategy):
    """
    AUROC strategy supporting:
      - binary single-label classification (num_classes=2)
      - multiclass single-label classification (num_classes>2)

    Binary:
      - accepts logits (B,2) or (B,) / (B,1)
      - uses positive-class probability

    Multiclass:
      - expects logits (B,C)
      - uses softmax probabilities
      - macro average by default
    """
    num_classes: int
    average: str = "macro"
    name: str = "auc"

    def __post_init__(self) -> None:
        if self.num_classes == 2:
            self._m = BinaryAUROC()
        else:
            self._m = MulticlassAUROC(num_classes=self.num_classes, average=self.average)

    def to(self, device: torch.device) -> "AUROCStrategy":
        self._m = self._m.to(device)
        return self

    def reset(self) -> None:
        self._m.reset()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        targets = targets.long()

        if self.num_classes == 2:
            # logits: (B,2) or (B,) / (B,1)
            if logits.ndim == 2 and logits.size(1) == 2:
                preds = torch.softmax(logits, dim=1)[:, 1]
            else:
                preds = torch.sigmoid(logits.view(-1))
            self._m.update(preds, targets.int())
        else:
            # logits must be (B,C)
            if logits.ndim != 2 or logits.size(1) != self.num_classes:
                raise ValueError(
                    f"Expected multiclass logits shape (B,{self.num_classes}), got {tuple(logits.shape)}"
                )
            preds = torch.softmax(logits, dim=1)  # (B,C) probabilities
            self._m.update(preds, targets)        # targets: (B,) in [0..C-1]

    def compute(self) -> float:
        v = self._m.compute()
        return float(v.detach().cpu().item())
