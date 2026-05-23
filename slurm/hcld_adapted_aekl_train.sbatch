#!/bin/bash
#SBATCH --job-name=harmonit_adapt_hcld_ae
#SBATCH --partition=gpu_lowpriority
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
PYTHON="${PYTHON:-/home/muqsitamir/miniconda3/envs/hcld/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/hcld_abide/adapted_hcld/aekl}"
RESUME_CKPT="${RESUME_CKPT:-${OUT_DIR}/checkpoints/model_latest.pt}"

cd "${REPO_DIR}"
mkdir -p logs "${OUT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

RESUME_ARGS=()
if [ -f "${RESUME_CKPT}" ]; then
  RESUME_ARGS=(--resume-checkpoint "${RESUME_CKPT}")
fi

"${PYTHON}" scripts/methods/hcld_train_autoencoder.py \
  --hcld-root /home/muqsitamir/repos/HCLD \
  --data-dir outputs/hcld_abide \
  --out-dir "${OUT_DIR}" \
  --batch-size 1 \
  --num-workers 4 \
  --epochs 30 \
  --save-every-steps 1000 \
  --val-every-steps 1000 \
  --export-recon-every-steps 1000 \
  --attention \
  --gradient-checkpointing \
  --amp-dtype bf16 \
  "${RESUME_ARGS[@]}"
