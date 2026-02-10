#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON=python
SCRIPT="src.cli.train"
EPOCHS=30
T=32
SEED_FILE="seeds.txt"
BASE_OUT_DIR="results"
LOG_DIR="logs/snn"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SEED_FILE}" ]]; then
    echo "ERROR: Seed file '${SEED_FILE}' not found."
    exit 1
fi

mapfile -t SEEDS < "${SEED_FILE}"

# -----------------------
# Ensure we run from project root
# -----------------------

cd "$PROJECT_ROOT"

echo "Running MHIST SNN experiments with seeds:"
printf "  %s\n" "${SEEDS[@]}"

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${BASE_OUT_DIR}/snn_seed${SEED}"
    LOG_FILE="${LOG_DIR}/snn_seed${SEED}.log"

    echo "=============================================="
    echo "Seed=${SEED}"
    echo "Out dir=${OUT_DIR}"
    echo "Log=${LOG_FILE}"
    echo "=============================================="
    echo "Running experiment..."
    ${PYTHON} -m ${SCRIPT} \
        --train-csv data/mhist_train_annotation.csv \
        --test-csv data/mhist_test_annotation.csv \
        --splits-dir data/splits/ \
        --model snn_resnet18 \
        --out-dir "${OUT_DIR}" \
        --path-col "Image Name" \
        --label-col "Majority Vote Label" \
        --root-dir /path/to/MHIST/images/ \
        --timesteps "${T}" \
        --epochs "${EPOCHS}" \
        --subset full \
        --lr 0.001 \
        --seed "${SEED}" \
        --pretrained \
        > "${LOG_FILE}" 2>&1
done

echo "All SNN seed runs completed."
