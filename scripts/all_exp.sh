#!/usr/bin/env bash
#SBATCH --job-name=snn_all_exp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=18:00:00
#SBATCH --cluster xxxx
#SBATCH --output=logs/slurm/xxxx_%j.out

set -euo pipefail

echo "=============================================="
echo "Experiments Started"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "=============================================="

# ---- environment ----
module load python/ondemand-jupyter-python3.11
source activate /path/to/envs/xxx/

# ---- directories ----
PROJECT_SCRIPTS="/path/to/scripts"
LOG_DIR="logs/slurm"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_SCRIPTS}"
echo "Running experiment scripts from: $(pwd)"

# ---- call experiment scripts ----
bash run_snn_exp.sh
bash run_snn_exp_pretrained.sh
bash run_cnn_train.sh
bash run_cnn_train_pretrained.sh

echo "=============================================="
echo "All experiments completed successfully."
echo "=============================================="
