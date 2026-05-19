#!/bin/bash
#SBATCH --job-name=harmonit_diff_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
PYTHON="${PYTHON:-/home/muqsitamir/miniconda3/envs/harmonit311/bin/python}"

cd "${REPO_DIR}"
mkdir -p logs outputs/harmonized/diffusion_img2img_nyu_smoke
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/methods/diffusion_img2img.py train \
  --out-dir outputs/harmonized/diffusion_img2img_nyu_smoke \
  --steps 4 \
  --batch-size 2 \
  --num-workers 0 \
  --base-channels 16 \
  --channel-mults 1 2 \
  --num-timesteps 100 \
  --log-every 1 \
  --save-every 4

"${PYTHON}" scripts/methods/diffusion_img2img.py export \
  --checkpoint outputs/harmonized/diffusion_img2img_nyu_smoke/checkpoints/model_latest.pt \
  --out-dir outputs/harmonized/diffusion_img2img_nyu_smoke \
  --batch-size 2 \
  --num-workers 0 \
  --base-channels 16 \
  --channel-mults 1 2 \
  --num-timesteps 100 \
  --ddim-steps 4 \
  --strength 0.25 \
  --no-qc
