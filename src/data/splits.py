"""
@file: splits.py
@brief: MHIST split generation utilities.
@description:
    Utilities for generating and persisting reproducible dataset splits for MHIST experiments.

    This module:
      1) Creates a fixed, independent validation split from the original MHIST training split.
      2) Defines the remaining samples as the post-validation training pool (base_train_idx).
      3) Samples balanced training subsets (e.g., N=100/200/400 per class) exclusively from the
         post-validation training pool for low-data experiments.
      4) Saves all split indices as .npy files along with a meta.json file containing the split
         specification (seed, validation fraction, subset sizes) and class counts for bookkeeping.
      5) Provides helper functions to load MHIST labels from a CSV and map string labels (HP/SSA)
         to integer class IDs.

    Notes:
      - All indices are relative to the original MHIST *training* split. The MHIST test split is
        intended to remain unchanged and should be used separately for final evaluation.
      - The --force behavior (if used by a CLI wrapper) prevents accidental overwrite of existing
        split artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np


# -----------------------------
# Split logic (kept local for script simplicity)
# If you prefer: move these into src/data/splits.py and import them here.
# -----------------------------

@dataclass(frozen=True)
class SplitSpec:
    seed: int
    val_fraction: float
    subsets_per_class: Tuple[int, ...]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _assert_wont_overwrite(out_dir: str, force: bool) -> None:
    if force:
        return
    if os.path.exists(out_dir) and any(os.listdir(out_dir)):
        raise FileExistsError(
            f"Output directory '{out_dir}' is not empty. "
            f"Refusing to overwrite. Use --force to overwrite."
        )


def _save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def generate_and_save_splits(
    out_dir: str,
    labels: np.ndarray,
    spec: SplitSpec,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """
    labels: array of shape (N_train,) for the ORIGINAL MHIST train split, values in {0,1}
    Returns metadata dict (also written to meta.json).
    """
    _assert_wont_overwrite(out_dir, force)
    _ensure_dir(out_dir)

    labels = np.asarray(labels).astype(int)
    n = len(labels)

    classes = sorted([int(c) for c in np.unique(labels).tolist()])
    if classes != [0, 1]:
        # still allow, but warn in meta
        pass

    rng = np.random.default_rng(spec.seed)

    # --- create val split from ORIGINAL train indices ---
    all_idx = np.arange(n)
    rng.shuffle(all_idx)

    split_point = int((1.0 - spec.val_fraction) * n)
    base_train_idx = all_idx[:split_point]
    val_idx = all_idx[split_point:]

    # Balanced subsets are sampled ONLY from base_train_idx
    base_labels = labels[base_train_idx]

    subset_files: Dict[str, str] = {}
    for k in spec.subsets_per_class:
        chosen_parts: List[np.ndarray] = []
        for c in classes:
            class_pool = base_train_idx[base_labels == c]
            if len(class_pool) < k:
                raise ValueError(
                    f"Not enough samples for class {c} to draw {k} examples. "
                    f"Available in base_train_idx: {len(class_pool)}."
                )
            chosen_parts.append(rng.choice(class_pool, size=k, replace=False))

        subset_idx = np.concatenate(chosen_parts)
        rng.shuffle(subset_idx)

        fname = f"subset_{k}_per_class_idx.npy"
        np.save(os.path.join(out_dir, fname), subset_idx)
        subset_files[str(k)] = fname

    np.save(os.path.join(out_dir, "base_train_idx.npy"), base_train_idx)
    np.save(os.path.join(out_dir, "val_idx.npy"), val_idx)

    # Meta: include helpful sanity stats
    meta: Dict[str, Any] = {
        "split_spec": asdict(spec),
        "n_original_train": int(n),
        "n_base_train": int(len(base_train_idx)),
        "n_val": int(len(val_idx)),
        "classes": classes,
        "subset_files": subset_files,
        "class_counts_original_train": {str(c): int((labels == c).sum()) for c in classes},
        "class_counts_base_train": {str(c): int((base_labels == c).sum()) for c in classes},
    }
    _save_json(os.path.join(out_dir, "meta.json"), meta)
    return meta


def load_labels_from_csv(train_csv: str, label_col: str) -> np.ndarray:
    """
    Load original MHIST labels from CSV file.
    Expects one row per image tile in the ORIGINAL MHIST train split.

    train_csv: path to a CSV containing at least {label_col}
    label_col: column name containing labels (0/1 or strings you will map)
    """

    labels: List[int] = []
    with open(train_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if label_col not in reader.fieldnames:
            raise ValueError(f"CSV missing label column '{label_col}'. Columns: {reader.fieldnames}")
        for row in reader:
            labels.append(row[label_col])

    return map_labels(np.array(labels))


def map_labels(raw: np.ndarray) -> np.ndarray:
    """
    Map raw labels to integer {0,1}.
    Adjust mapping here if your CSV uses different strings.
    """

    # string mapping
    s = np.array([str(x).strip() for x in raw], dtype=object)
    mapping = {
        "HP": 0,
        "SSA": 1,
    }

    out = np.empty(len(s), dtype=int)

    for i, v in enumerate(s):
        if v not in mapping:
            raise ValueError(f"Unrecognized label '{raw[i]}' at row {i}. Update map_labels().")
        out[i] = mapping[v]
    return out

def load_saved_splits(split_dir: str) -> Dict[str, np.ndarray]:
    """
    Loads the split artifacts produced by your setup/make_splits CLI.

    Returns:
      base_train_idx: indices into ORIGINAL MHIST train split (post-val training pool)
      val_idx:        indices into ORIGINAL MHIST train split
      subset_{k}_per_class_idx: indices into ORIGINAL MHIST train split
    """
    meta_path = os.path.join(split_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in split_dir: {split_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    out: Dict[str, np.ndarray] = {
        "base_train_idx": np.load(os.path.join(split_dir, "base_train_idx.npy")),
        "val_idx": np.load(os.path.join(split_dir, "val_idx.npy")),
    }

    for k, fname in meta["subset_files"].items():
        out[f"subset_{k}_per_class_idx"] = np.load(os.path.join(split_dir, fname))

    return out
