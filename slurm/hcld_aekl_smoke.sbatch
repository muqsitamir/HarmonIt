#!/bin/bash
#SBATCH --job-name=harmonit_hcld_ae_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
PYTHON="${PYTHON:-/home/muqsitamir/miniconda3/envs/hcld/bin/python}"

cd "${REPO_DIR}"
mkdir -p logs outputs/hcld_abide/aekl_smoke
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/methods/hcld_train_autoencoder.py \
  --hcld-root /home/muqsitamir/repos/HCLD \
  --data-dir outputs/hcld_abide \
  --out-dir outputs/hcld_abide/aekl_smoke \
  --batch-size 1 \
  --num-workers 0 \
  --max-steps 2 \
  --save-every-steps 2 \
  --val-every-steps 2
