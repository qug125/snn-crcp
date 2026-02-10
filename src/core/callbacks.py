# -*- coding: utf-8 -*-
"""
@file: callbacks.py
@description:
    Short description of what this script does.
"""
import math
import os
import torch
from typing import Any, Dict, Protocol, Optional

class Callback(Protocol):
    """Observer/Hook style callbacks for extensibility."""
    def on_train_begin(self, state: Dict[str, Any]) -> None: ...
    def on_epoch_end(self, state: Dict[str, Any]) -> None: ...
    def on_train_end(self, state: Dict[str, Any]) -> None: ...

class CSVLogger(Callback):
    def __init__(self, out_path: str):
        self.out_path = out_path
        self._initialized = False

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            f.write("epoch,train_loss,val_auc,lr\n")
        self._initialized = True

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        if not self._initialized:
            return
        epoch = state["epoch"]
        train_loss = state["train_loss"]
        val_auc = state.get("val_auc", float("nan"))
        lr = state["lr"]
        with open(self.out_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_auc:.6f},{lr:.8f}\n")

    def on_train_end(self, state: Dict[str, Any]) -> None:
        return


class BestCheckpoint(Callback):
    """Save best model by validation AUC."""
    def __init__(self, out_dir: str, metric_key: str = "val_auc", higher_is_better: bool = True):
        self.out_dir = out_dir
        self.metric_key = metric_key
        self.higher_is_better = higher_is_better
        self.best = -float("inf") if higher_is_better else float("inf")
        os.makedirs(out_dir, exist_ok=True)

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        return

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        metric = state.get(self.metric_key, None)
        if metric is None or (isinstance(metric, float) and math.isnan(metric)):
            return

        improved = metric > self.best if self.higher_is_better else metric < self.best
        if improved:
            self.best = metric
            path = os.path.join(self.out_dir, "best.pt")
            torch.save(
                {
                    "epoch": state["epoch"],
                    "model_state": state["model"].state_dict(),
                    "optimizer_state": state["optimizer"].state_dict(),
                    self.metric_key: metric,
                },
                path,
            )

    def on_train_end(self, state: Dict[str, Any]) -> None:
        return
