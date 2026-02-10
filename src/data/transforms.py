"""
@file: transforms.py
@brief: Image preprocessing and augmentation utilities for MHIST experiments.
@description:
    Defines standardized image transformation pipelines for training, validation,
    and testing in MHIST colorectal histopathology classification experiments.

    This module provides:
      - A minimal, conservative training transform consisting of spatial resizing,
        stochastic horizontal and vertical flips, tensor conversion, and channel-wise
        normalization.
      - A deterministic evaluation transform for validation and test sets that applies
        only resizing, tensor conversion, and normalization.

    The augmentation strategy is intentionally limited, as the original MHIST benchmark
    does not specify explicit data augmentation procedures. All transforms are designed
    to be applied consistently across spiking and non-spiking models to ensure fair
    architectural comparison under controlled experimental conditions.
"""


from __future__ import annotations

from typing import Tuple, Optional
from torchvision import transforms


class TransformFactory:
    @staticmethod
    def create(model_type: str, train: bool = True, resize=(224,224)):
        if model_type == "resnet18":
            return TransformFactory._cnn_transforms(train, resize)
        elif model_type == "snn_resnet18":
            return TransformFactory._snn_poisson_transforms(train, resize)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


    @staticmethod
    def _cnn_transforms(train: bool, resize: Tuple[int, int]):
        ops = [transforms.Resize(resize)]
        if train:
            ops += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ]

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        return transforms.Compose(ops)

    @staticmethod
    def _snn_poisson_transforms(train: bool, resize: Tuple[int, int]):
        print("SNN Transforms")
        ops = [transforms.Resize(resize)]
        if train:
            ops += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ]

        ops += [
            transforms.ToTensor(),
        ]
        return transforms.Compose(ops)
