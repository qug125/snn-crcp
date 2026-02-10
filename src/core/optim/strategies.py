# -*- coding: utf-8 -*-
"""
@file: strategies.py
@description:
    Short description of what this script does.
"""

from dataclasses import dataclass
from typing import Dict, Protocol, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.core.config import TrainConfig


class OptimizerStrategy(Protocol):
    name: str
    def build(self, model: torch.nn.Module, cfg: TrainConfig) -> Optimizer: ...


class SchedulerStrategy(Protocol):
    name: str
    # step_mode tells Trainer whether to call step() per batch or per epoch
    step_mode: str  # "batch" | "epoch"
    def build(self, optimizer: Optimizer, cfg: TrainConfig, steps_per_epoch: int) -> Optional[_LRScheduler]: ...


@dataclass(frozen=True)
class AdamWStrategy:
    name: str = "adamw"
    def build(self, model: torch.nn.Module, cfg: TrainConfig) -> Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


@dataclass(frozen=True)
class SGDStrategy:
    name: str = "sgd"
    def build(self, model: torch.nn.Module, cfg: TrainConfig) -> Optimizer:
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=getattr(cfg, "momentum", 0.9),
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )


@dataclass(frozen=True)
class CosineAnnealStrategy:
    name: str = "cosine"
    step_mode: str = "batch"
    def build(self, optimizer: Optimizer, cfg: TrainConfig, steps_per_epoch: int) -> _LRScheduler:
        total_steps = cfg.epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-06)


@dataclass(frozen=True)
class StepLRStrategy:
    name: str = "step"
    step_mode: str = "epoch"
    def build(self, optimizer: Optimizer, cfg: TrainConfig, steps_per_epoch: int) -> _LRScheduler:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)


@dataclass(frozen=True)
class ExponentialLRStrategy:
    """
    Multiplicative / exponential learning rate decay:
        lr_t = lr_0 * gamma^t

    Matches MHIST-style 'exponential decay'.
    """
    name: str = "exp"
    step_mode: str = "epoch"

    def build(
        self,
        optimizer: Optimizer,
        cfg: TrainConfig,
        steps_per_epoch: int,
    ) -> _LRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.lr_decay_gamma,
        )


OPTIMIZER_STRATEGIES: Dict[str, OptimizerStrategy] = {
    "adamw": AdamWStrategy(),
    "sgd": SGDStrategy(),
}

SCHEDULER_STRATEGIES: Dict[str, SchedulerStrategy] = {
    "cosine": CosineAnnealStrategy(),
    "step": StepLRStrategy(),
    "exp": ExponentialLRStrategy()
}
