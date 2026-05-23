#!/bin/bash
#SBATCH --job-name=harmonit_adapt_hcld_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
PYTHON="${PYTHON:-/home/muqsitamir/miniconda3/envs/hcld/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/hcld_abide/adapted_hcld/aekl_smoke}"

cd "${REPO_DIR}"
mkdir -p logs "${OUT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/methods/hcld_train_autoencoder.py \
  --hcld-root /home/muqsitamir/repos/HCLD \
  --data-dir outputs/hcld_abide \
  --out-dir "${OUT_DIR}" \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 300 \
  --save-every-steps 100 \
  --val-every-steps 100 \
  --export-recon-every-steps 100 \
  --attention \
  --gradient-checkpointing \
  --amp-dtype bf16
