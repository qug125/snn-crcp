# -*- coding: utf-8 -*-
"""
@file: infer.py
@brief: CLI entrypoint for running inference on MHIST using a saved checkpoint.
@description:
    Loads a trained model checkpoint and runs inference/evaluation on MHIST
    (validation or test split). Supports ANN and SpikingJelly SNN models,
    computes AUROC via torchmetrics, and optionally saves logits.

    PATCH:
      - Passes --timesteps into wrap_model_as_strategy() so SNN inference uses the same T as training.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.utils.seed import seed_everything
from src.utils.io import ensure_dir

from src.core.config import ModelConfig
from src.core.metrics import AUROCStrategy

from src.data.dataset import MHISTDataset
from src.data.transforms import TransformFactory

from src.models.registry import MODEL_BUILDERS, build_model
from src.models.strategies.utils import wrap_model_as_strategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference/evaluation on MHIST using a saved checkpoint.")

    # Data
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True)
    p.add_argument("--root-dir", type=str, default=None)

    # Added previously (keep)
    p.add_argument("--path-col", type=str, default="path")
    p.add_argument("--label-col", type=str, default="label")

    # Splits
    p.add_argument("--splits-dir", type=str, required=True)
    p.add_argument("--split", type=str, choices=["val", "test"], required=True)

    # Model
    p.add_argument("--model", type=str, required=True, choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--input-channels", type=int, default=3)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)

    # Keep timesteps flag (SNN only)
    p.add_argument("--timesteps", type=int, default=4)

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True)

    # Runtime
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--save-logits", action="store_true")

    return p.parse_args()


def load_val_indices(splits_dir: str) -> np.ndarray:
    return np.load(os.path.join(splits_dir, "val_idx.npy")).astype(int)


@torch.no_grad()
def run_inference(
    model_strategy,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model = model_strategy.model().to(device)
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model_strategy.reset_state()
        logits = model(x)

        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    return (
        torch.cat(all_logits, dim=0).numpy(),
        torch.cat(all_labels, dim=0).numpy(),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    eval_tf = TransformFactory.create(args.model, train=False, resize=(args.resize, args.resize))

    # Datasets
    train_eval = MHISTDataset(
        csv_path=args.train_csv,
        root_dir=args.root_dir,
        path_col=args.path_col,
        label_col=args.label_col,
        transform=eval_tf,
    )
    test_ds = MHISTDataset(
        csv_path=args.test_csv,
        root_dir=args.root_dir,
        path_col=args.path_col,
        label_col=args.label_col,
        transform=eval_tf,
    )

    if args.split == "val":
        val_idx = load_val_indices(args.splits_dir)
        ds = Subset(train_eval, val_idx.tolist())
        split_name = "val"
    else:
        ds = test_ds
        split_name = "test"

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model
    model_cfg = ModelConfig(
        name=args.model,
        num_classes=args.num_classes,
        input_channels=args.input_channels,
        pretrained=args.pretrained,
        dropout_p=args.dropout,
        timesteps=args.timesteps,
    )

    model = build_model(model_cfg)

    # ✅ PATCH: pass timesteps through wrapper so SNN inference matches training T
    model_strategy = wrap_model_as_strategy(
        model_name=args.model,
        model=model,
        timesteps=args.timesteps,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state" not in ckpt:
        raise KeyError("Checkpoint missing 'model_state'. Expected BestCheckpoint format.")
    model_strategy.model().load_state_dict(ckpt["model_state"], strict=True)

    # Inference
    logits, labels = run_inference(model_strategy, loader, device)

    # Metric (compute on CPU for simplicity)
    metrics = AUROCStrategy(num_classes=args.num_classes)
    metrics.reset()
    metrics.update(torch.from_numpy(logits), torch.from_numpy(labels))
    auc = metrics.compute()

    print(f"{split_name} auc: {auc:.6f}")

    # Save logits if requested
    if args.save_logits:
        if args.out_dir is None:
            raise ValueError("--save-logits requires --out-dir")

        ensure_dir(args.out_dir)
        out_path = os.path.join(args.out_dir, f"{split_name}_logits.npz")
        np.savez(out_path, logits=logits, labels=labels)
        print(f"[OK] Saved logits to: {out_path}")


if __name__ == "__main__":
    main()
