#!/usr/bin/env bash
set -e

# -----------------------
# Resolve project root
# -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


PYTHON=python
MODULE="src.cli.inference"

SEED_FILE="seeds.txt"
RESULTS_DIR="results"
LOG_DIR="logs/inference/snn_pretrained"


if [[ ! -f "${SEED_FILE}" ]]; then
  echo "ERROR: Seed file not found: ${SEED_FILE}"
  exit 1
fi

mapfile -t SEEDS < "${SEED_FILE}"

echo "Running inference for seeds:"
printf "  %s\n" "${SEEDS[@]}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="${RESULTS_DIR}/snn_seed${SEED}_pretrained"
  CKPT="${RUN_DIR}/checkpoints/best.pt"
  LOG_FILE="${LOG_DIR}/snn_seed${SEED}_pretrained.log"

  if [[ ! -f "${CKPT}" ]]; then
    echo "WARNING: checkpoint not found for seed ${SEED}: ${CKPT}"
    echo "Skipping."
    continue
  fi

  echo "=============================================="
  echo "Seed=${SEED}"
  echo "Checkpoint=${CKPT}"
  echo "=============================================="

  ${PYTHON} -m ${MODULE} \
    --train-csv data/mhist_train_annotation.csv \
    --test-csv data/mhist_test_annotation.csv \
    --splits-dir data/splits/ \
    --model snn_resnet18 \
    --timesteps 8 \
    --out-dir "${RUN_DIR}" \
    --path-col "Image Name" \
    --label-col "Majority Vote Label" \
    --root-dir /path/to/MHIST/images/ \
    --split test \
    --checkpoint "${CKPT}" \
    --pretrained \
    > "${LOG_FILE}" 2>&1
done

echo "Inference complete."
