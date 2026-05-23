#!/bin/bash
#SBATCH --job-name=harmonit_adapt_hcld_ldm_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
PYTHON="${PYTHON:-/home/muqsitamir/miniconda3/envs/hcld/bin/python}"
AE_CKPT="${AE_CKPT:-outputs/hcld_abide/adapted_hcld/aekl/checkpoints/model_latest.pt}"
OUT_DIR="${OUT_DIR:-outputs/hcld_abide/adapted_hcld/ldm_smoke}"

cd "${REPO_DIR}"
mkdir -p logs "${OUT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/methods/hcld_train_ldm.py \
  --hcld-root /home/muqsitamir/repos/HCLD \
  --data-dir outputs/hcld_abide \
  --ae-checkpoint "${AE_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 25 \
  --save-every-steps 25 \
  --val-every-steps 25 \
  --amp-dtype bf16
