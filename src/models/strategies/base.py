# -*- coding: utf-8 -*-
"""
@file: base.py
@description:
    Short description of what this script does.
"""

import torch.nn as nn
from typing import Protocol

class ModelStrategy(Protocol):
    def model(self) -> nn.Module: ...
    def reset_state(self) -> None: ...


