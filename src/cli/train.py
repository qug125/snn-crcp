import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.utils.seed import seed_everything
from src.utils.io import ensure_dir

from src.core.trainer import Trainer
from src.core.config import TrainConfig, RunConfig, ModelConfig
from src.core.metrics import AUROCStrategy

from src.data.dataset import MHISTDataset
from src.data.transforms import TransformFactory

from src.models.registry import MODEL_BUILDERS, build_model
from src.models.strategies.utils import wrap_model_as_strategy

from src.core.loss import CrossEntropyLossStrategy, SpikeRateMSELossStrategy

from src.core.callbacks import BestCheckpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ANN/SNN models on MHIST with fixed splits.")

    # -------- Data / Splits --------
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True)  # optional usage; kept for completeness
    p.add_argument("--root-dir", type=str, default=None)
    p.add_argument("--splits-dir", type=str, required=True)
    p.add_argument("--subset", type=str, default="full", choices=["full", "100", "200", "400"])
    p.add_argument("--resize", type=int, default=224)

    # CSV columns
    p.add_argument("--path-col", type=str, default="path")
    p.add_argument("--label-col", type=str, default="label")

    # -------- Model --------
    p.add_argument("--model", type=str, required=True, choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--input-channels", type=int, default=3)
    p.add_argument("--pretrained", action="store_true", default=False)
    p.add_argument("--dropout", type=float, default=0.0)

    # SNN-specific (ignored by ANN builder if unused)
    p.add_argument("--timesteps", type=int, default=4)

    # -------- Training --------
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.91,
        help="Gamma for exponential LR decay (lr_t = lr_0 * gamma^t)",
    )

    # -------- Runtime --------
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--experiment-name", type=str, default="mhist_run")

    return p.parse_args()


def load_split_indices(splits_dir: str, subset: str) -> Tuple[np.ndarray, np.ndarray]:
    base_train = np.load(os.path.join(splits_dir, "base_train_idx.npy")).astype(int)
    val_idx = np.load(os.path.join(splits_dir, "val_idx.npy")).astype(int)

    if subset == "full":
        train_idx = base_train
    else:
        train_idx = np.load(os.path.join(splits_dir, f"subset_{subset}_per_class_idx.npy")).astype(int)

    return train_idx, val_idx


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    # ------------------------
    # Configs
    # ------------------------
    model_cfg = ModelConfig(
        name=args.model,
        num_classes=args.num_classes,
        input_channels=args.input_channels,
        pretrained=args.pretrained,
        dropout_p=args.dropout,
        timesteps=args.timesteps,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        grad_clip_norm=args.grad_clip,
        amp=args.amp,
        device=args.device,
        seed=args.seed,
    )

    run_cfg = RunConfig(
        out_dir=args.out_dir,
        experiment_name=args.experiment_name,
        # optional: log_file=os.path.join(args.out_dir, "train.log"),
    )

    # ------------------------
    # Transforms
    # ------------------------
    train_tf = TransformFactory.create(args.model, train=True)
    eval_tf = TransformFactory.create(args.model, train=False)

    # ------------------------
    # Datasets (two train-base instances so val is deterministic)
    # ------------------------
    train_base_aug = MHISTDataset(
        csv_path=args.train_csv,
        root_dir=args.root_dir,
        path_col=args.path_col,
        label_col=args.label_col,
        transform=train_tf,
    )
    train_base_eval = MHISTDataset(
        csv_path=args.train_csv,
        root_dir=args.root_dir,
        path_col=args.path_col,
        label_col=args.label_col,
        transform=eval_tf,
    )

    # Optional: build test dataset for later evaluation if you want
    _ = MHISTDataset(
        csv_path=args.test_csv,
        root_dir=args.root_dir,
        path_col=args.path_col,
        label_col=args.label_col,
        transform=eval_tf,
    )

    # ------------------------
    # Apply fixed splits
    # ------------------------
    train_idx, val_idx = load_split_indices(args.splits_dir, args.subset)

    if np.intersect1d(train_idx, val_idx).size > 0:
        raise RuntimeError("Train/val overlap detected in split indices. Check your split artifacts.")

    train_ds = Subset(train_base_aug, train_idx.tolist())
    val_ds = Subset(train_base_eval, val_idx.tolist())

    # ------------------------
    # DataLoaders
    # ------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------
    # Build model via ModelBuilder registry and wrap into trainer ModelStrategy
    # ------------------------
    model = build_model(model_cfg)
    model_strategy = wrap_model_as_strategy(model_name=args.model, model=model, timesteps=args.timesteps)

    # Loss + metrics
    loss_fn = CrossEntropyLossStrategy()

    if args.model == "snn_resnet18":
        print("[OK] Using MSE Loss for SNN")
        loss_fn = SpikeRateMSELossStrategy()
    else:
        print("[OK] Using CE Loss for CNN")


    metrics = AUROCStrategy(num_classes=args.num_classes)  # metrics.name should be "auc"

    # ------------------------
    # Callbacks (Best checkpoint by val_auc)
    # ------------------------
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    callbacks = [
        BestCheckpoint(out_dir=ckpt_dir, metric_key="val_auc", higher_is_better=True)
    ]

    # ------------------------
    # Train
    # ------------------------
    trainer = Trainer(
        model_strategy=model_strategy,
        loss_strategy=loss_fn,
        metrics_strategy=metrics,
        train_cfg=train_cfg,
        run_cfg=run_cfg,
        callbacks=callbacks,
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
