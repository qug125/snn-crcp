from dataclasses import asdict
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

from src.core.config import TrainConfig, RunConfig
from src.core.protocols import LossStrategy, Callback
from src.models.strategies.base import ModelStrategy
from src.core.metrics import MetricsStrategy

# strategy-based optimizer/scheduler
from src.core.optim.strategies import OPTIMIZER_STRATEGIES, SCHEDULER_STRATEGIES


class Trainer:
    """
    Unified trainer for ANN and SpikingJelly SNN models.

    Key design points:
      - Expects a trainer-facing ModelStrategy:
          * model() -> nn.Module
          * reset_state() -> None   (no-op for ANN; reset_net for SpikingJelly)
      - Builds optimizer/scheduler based on cfg via strategy registries
      - Supports scheduler step modes:
          * "batch"  (e.g., cosine anneal per step)
          * "epoch"  (e.g., step LR, exponential LR)
      - Computes validation metric via MetricsStrategy (torchmetrics-backed)
      - Exposes both:
          * state["val_metric"]
          * state[f"val_{metric_name}"] (e.g., val_auc)
        so callbacks can monitor stable keys.
    """

    def __init__(
        self,
        model_strategy: ModelStrategy,
        loss_strategy: LossStrategy,
        metrics_strategy: MetricsStrategy,
        train_cfg: TrainConfig,
        run_cfg: RunConfig,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.ms = model_strategy
        self.loss_fn = loss_strategy
        self.metrics = metrics_strategy
        self.train_cfg = train_cfg
        self.run_cfg = run_cfg
        self.callbacks = callbacks or []

        self.device = torch.device(train_cfg.device)

        # Model from strategy (already built in CLI)
        self.model = self.ms.model().to(self.device)

        # Optimizer / scheduler created in fit()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scheduler_step_mode: Optional[str] = None  # "batch" | "epoch" | None

        # AMP
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(train_cfg.amp) and self.device.type == "cuda"
        )

        # Move metrics to device if supported
        if hasattr(self.metrics, "to"):
            self.metrics = self.metrics.to(self.device)

        # Logging + run dir
        ensure_dir(self.run_cfg.out_dir)
        log_file = getattr(self.run_cfg, "log_file", None)
        self.logger = get_logger(
            name=f"trainer:{getattr(self.run_cfg, 'experiment_name', 'run')}",
            log_file=log_file,
        )

    # -----------------------
    # Public API
    # -----------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        seed_everything(self.train_cfg.seed)

        self.optimizer = self._build_optimizer()
        self.scheduler, self.scheduler_step_mode = self._build_scheduler(steps_per_epoch=self.train_cfg.epochs)

        # Persist configs
        save_json(
            path=f"{self.run_cfg.out_dir}/config.json",
            obj={"train": asdict(self.train_cfg), "run": asdict(self.run_cfg)},
        )

        metric_name = getattr(self.metrics, "name", "metric")

        state: Dict[str, Any] = {
            "model": self.model,
            "optimizer": self.optimizer,
            "epoch": 0,
            "train_loss": float("nan"),
            "val_metric": float("nan"),
            f"val_{metric_name}": float("nan"),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        for cb in self.callbacks:
            cb.on_train_begin(state)

        for epoch in range(1, self.train_cfg.epochs + 1):
            state["epoch"] = epoch

            lr_before = self.optimizer.param_groups[0]["lr"]
            train_loss = self._train_one_epoch(train_loader)
            val_metric = self._evaluate_metric(val_loader)
            lr_after = self.optimizer.param_groups[0]["lr"]

            state["train_loss"] = train_loss
            state["val_metric"] = val_metric
            state[f"val_{metric_name}"] = val_metric  # e.g., val_auc
            state["lr"] = lr_after

            self.logger.info(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_{metric_name}={val_metric:.6f} lr={lr_before:.8e} lr_next={lr_after:.8e}"
            )

            for cb in self.callbacks:
                cb.on_epoch_end(state)

        for cb in self.callbacks:
            cb.on_train_end(state)

        # Save final summary
        save_json(
            path=f"{self.run_cfg.out_dir}/final_metrics.json",
            obj={
                "final_train_loss": float(state["train_loss"]),
                f"final_val_{metric_name}": float(state[f"val_{metric_name}"]),
            },
        )

        return state

    # -----------------------
    # Internal builders
    # -----------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        key = str(self.train_cfg.optimizer).lower()
        if key not in OPTIMIZER_STRATEGIES:
            raise ValueError(f"Unknown optimizer '{self.train_cfg.optimizer}'. Options: {list(OPTIMIZER_STRATEGIES)}")
        return OPTIMIZER_STRATEGIES[key].build(self.model, self.train_cfg)

    def _build_scheduler(self, steps_per_epoch: int) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], Optional[str]]:
        key = str(self.train_cfg.scheduler).lower()
        if key in {"none", "", "null"}:
            return None, None
        if key not in SCHEDULER_STRATEGIES:
            raise ValueError(f"Unknown scheduler '{self.train_cfg.scheduler}'. Options: {list(SCHEDULER_STRATEGIES)}")
        strat = SCHEDULER_STRATEGIES[key]
        sched = strat.build(self.optimizer, self.train_cfg, steps_per_epoch)
        return sched, getattr(strat, "step_mode", None)

    # -----------------------
    # Train / eval
    # -----------------------

    def _train_one_epoch(self, loader: DataLoader) -> float:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call fit() first.")

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Reset spiking state if needed (no-op for ANN)
            self.ms.reset_state()

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(x)
                loss = self.loss_fn(logits, y)

            self.scaler.scale(loss).backward()

            if getattr(self.train_cfg, "grad_clip_norm", None) is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # # Per-batch scheduler step (e.g., cosine)
            # if self.scheduler is not None and self.scheduler_step_mode == "batch":
            #     self.scheduler.step()

            total_loss += float(loss.item())
            n_batches += 1

        # Per-epoch scheduler step (e.g., step LR, exponential LR)
        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def _evaluate_metric(self, loader: DataLoader) -> float:
        self.model.eval()
        self.metrics.reset()

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Reset spiking state if needed (no-op for ANN)
            self.ms.reset_state()

            logits = self.model(x)
            self.metrics.update(logits, y)

        return self.metrics.compute()