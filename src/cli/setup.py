"""
@file: setup.py
@brief: Command-line interface for generating fixed MHIST dataset splits.
@description:
    Command-line utility for generating and persisting reproducible dataset splits for MHIST
    experiments. This script serves as a thin orchestration layer over the split-generation
    utilities defined in src.data.splits.

    Specifically, this script:
      1) Loads class labels for the ORIGINAL MHIST training split from a user-provided CSV file.
      2) Constructs a SplitSpec defining the random seed, validation fraction, and balanced
         subset sizes per class.
      3) Generates a fixed validation split from the original training data.
      4) Defines the remaining samples as the post-validation training pool.
      5) Samples balanced training subsets (e.g., N=100/200/400 per class) exclusively from the
         post-validation training pool.
      6) Persists all split indices (.npy) and split metadata (meta.json) to disk for reuse across
         experiments.

    All generated indices reference the original MHIST training split. The MHIST test split is
    not modified and should be used separately for final model evaluation.

    This script is intended to be run once per random seed and dataset configuration. Subsequent
    training runs should load the persisted split files rather than regenerating splits.

    Usage:
        python -m src.cli.setup \
            --train-csv /path/to/mhist_train.csv \
            --label-col label \
            --out-dir splits/mhist/seed42 \
            --seed 42 \
            --val-fraction 0.2 \
            --subsets 100 200 400

    To overwrite an existing split directory, add the --force flag.

    Example:
        python -m src.cli.setup \
            --train-csv data/mhist_train.csv \
            --out-dir splits/mhist/seed42 \
            --force
"""

import argparse
import os
from src.data.splits import SplitSpec, generate_and_save_splits, load_labels_from_csv

def main() -> None:
    p = argparse.ArgumentParser(description="Generate fixed MHIST train/val splits and balanced training subsets.")
    p.add_argument("--train-csv", type=str, required=True, help="CSV for ORIGINAL MHIST train split (one row per tile).")
    p.add_argument("--label-col", type=str, default="label", help="Column name for label in CSV (default: label).")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory to write split files.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    p.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of ORIGINAL train used for validation.")
    p.add_argument("--subsets", type=int, nargs="+", default=[100, 200, 400], help="Balanced subset sizes per class.")
    p.add_argument("--force", action="store_true", help="Overwrite existing files in out-dir.")
    args = p.parse_args()

    labels = load_labels_from_csv(args.train_csv, args.label_col)

    spec = SplitSpec(
        seed=args.seed,
        val_fraction=float(args.val_fraction),
        subsets_per_class=tuple(int(x) for x in args.subsets),
    )

    meta = generate_and_save_splits(
        out_dir=args.out_dir,
        labels=labels,
        spec=spec,
        force=args.force,
    )

    print(f"[OK] Wrote splits to: {args.out_dir}")
    print(f"  base_train_idx: {os.path.join(args.out_dir, 'base_train_idx.npy')}")
    print(f"  val_idx:        {os.path.join(args.out_dir, 'val_idx.npy')}")
    for k, fname in meta["subset_files"].items():
        print(f"  subset_{k}_per_class_idx: {os.path.join(args.out_dir, fname)}")
    print(f"  meta:           {os.path.join(args.out_dir, 'meta.json')}")


if __name__ == "__main__":
    main()
