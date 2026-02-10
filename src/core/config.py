# -*- coding: utf-8 -*-
"""
@file: config.py.py
@description:
    Short description of what this script does.
"""

import torch
from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class DataConfig:
    num_workers: int = 4
    pin_memory: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # "sgd" or "adamw"
    scheduler: str = "cosine"  # "cosine" or "step" or "none"
    step_size: int = 10
    gamma: float = 0.1
    warmup_epochs: int = 0
    seed: int = 42
    amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr_decay_gamma: float = 0.9

@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 2
    pretrained: bool = True
    dropout_p: float = 0.0  # set >0 if you want
    input_channels: int = 3  # MHIST is RGB
    timesteps: int = 1


@dataclass(frozen=True)
class RunConfig:
    out_dir: str = "./runs/resnet18_mhist"
    experiment_name: str = "resnet18_ann"
    save_best: bool = True
    metric_name: str = "auc"  # MHIST-aligned
