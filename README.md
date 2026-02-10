# SNN vs CNN for Binary Serrated Colorectal Polyp Classification (HP versus SSA)

A comparative study of Spiking Neural Networks (SNNs) and Convolutional Neural Networks (CNNs) for histopathological image classification using the DHMC MHIST dataset.

## Overview

This project implements and compares ResNet18-based architectures in both traditional CNN and SNN variants for binary serrated colorectal polyp subtype classification. The codebase supports training from scratch or using pretrained weights, with comprehensive experiment management and evaluation tools.

## Features

- **Dual Architecture Support**: ResNet18 CNN and SNN implementations
- **Flexible Training**: Support for training from scratch or fine-tuning pretrained models
- **Reproducible Experiments**: Multi-seed training with fixed data splits


## Project Structure

```
.
├── scripts/                    # Experiment and inference scripts
│   ├── all_exp.sh             # Run all experiments (SLURM)
│   ├── run_snn_exp.sh         # Train SNN models
│   ├── run_snn_exp_pretrained.sh
│   ├── run_cnn_exp.sh        # Train CNN models
│   ├── run_cnn_pretrained.sh
│   ├── snn_inference.sh      # Inference scripts
│   ├── cnn_inference.sh
│   └── seeds.txt              # Random seeds for reproducibility
├── src/
│   ├── cli/                   # Command-line interfaces
│   │   ├── train.py           # Training entry point
│   │   ├── inference.py      # Inference entry point
│   │   └── setup.py          # Setup utilities
│   ├── core/                  # Core training components
│   │   ├── trainer.py        # Main trainer class
│   │   ├── config.py         # Configuration dataclasses
│   │   ├── callbacks.py      # Training callbacks
│   │   ├── loss.py           # Loss functions
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── optim/            # Optimizers and schedulers
│   ├── data/                  # Data handling
│   │   ├── dataset.py        # MHIST dataset loader
│   │   ├── splits.py         # Data splitting utilities
│   │   └── transforms.py     # Image transformations
│   ├── models/                # Model architectures
│   │   ├── registry.py       # Model registry
│   │   ├── builders/         # Model builders
│   │   ├── factories/        # Model factories
│   │   └── strategies/       # Model wrappers for training
│   └── utils/                 # Utility functions
├── data/                      # Data directory (not included)
│   ├── mhist_train_annotation.csv
│   ├── mhist_test_annotation.csv
│   └── splits/               # Preprocessed data splits
├── results/                   # Experiment results
└── logs/                      # Training logs
```


## Usage

### Training

#### Train SNN Model
```bash
python -m src.cli.train \
    --train-csv data/mhist_train_annotation.csv \
    --test-csv data/mhist_test_annotation.csv \
    --splits-dir data/splits/ \
    --model snn_resnet18 \
    --out-dir results/snn_experiment \
    --path-col "Image Name" \
    --label-col "Majority Vote Label" \
    --root-dir /path/to/MHIST/images/ \
    --timesteps 32 \
    --epochs 30 \
    --lr 0.001 \
    --seed 42
```

#### Train CNN Model
```bash
python -m src.cli.train \
    --train-csv data/mhist_train_annotation.csv \
    --test-csv data/mhist_test_annotation.csv \
    --splits-dir data/splits/ \
    --model resnet18 \
    --out-dir results/cnn_experiment \
    --path-col "Image Name" \
    --label-col "Majority Vote Label" \
    --root-dir /path/to/MHIST/images/ \
    --epochs 30 \
    --lr 0.001 \
    --seed 42
```

### Inference

Run inference on trained models:

```bash
python -m src.cli.inference \
    --train-csv data/mhist_train_annotation.csv \
    --test-csv data/mhist_test_annotation.csv \
    --splits-dir data/splits/ \
    --model snn_resnet18 \
    --timesteps 8 \
    --out-dir results/snn_seed42 \
    --path-col "Image Name" \
    --label-col "Majority Vote Label" \
    --root-dir /path/to/MHIST/images/ \
    --split test \
    --checkpoint results/snn_seed42/checkpoints/best.pt
```

Or use the batch inference scripts:

```bash
# Run inference for all SNN models
bash scripts/snn_inference.sh

# Run inference for all CNN models
bash scripts/cnn_inference.sh
```

## Configuration

### Model Options
- `resnet18`: Standard ResNet18 CNN
- `snn_resnet18`: Spiking ResNet18

### Training Parameters
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay for regularization (default: 1e-4)
- `--optimizer`: Optimizer choice (default: adamw)
- `--scheduler`: LR scheduler (default: cosine)
- `--grad-clip`: Gradient clipping value (optional)
- `--amp`: Enable automatic mixed precision

### SNN-Specific Parameters
- `--timesteps`: Number of timesteps for SNN simulation (default: 4)

### Data Parameters
- `--subset`: Training subset size (full, 100, 200, 400)
- `--resize`: Image resize dimension (default: 224)
- `--num-workers`: DataLoader workers (default: 4)

## Reproducibility

The project uses fixed random seeds for reproducible results. The `scripts/seeds.txt` file contains 10 predefined seeds:
```
123456
987654
20260116
42
314159
271828
777777
888888
999999
13579
```

All experiments scripts automatically iterate through these seeds to ensure statistical robustness.


## Citation

If you use this code in your research, please cite:

```bibtex
@article{2026snnrcrp,
  title={Comparative Analysis of Spiking and Convolutional Neural Networks for Colorectal Polyp Classification},
  author={Nickolas Littlefield, Riyue Bao, Rong Xia, Qiangqiang Gu},
  journal={MedRxiv},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
