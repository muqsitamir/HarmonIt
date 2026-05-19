#!/bin/bash
#SBATCH --job-name=harmonit_diff_nyu
#SBATCH --partition=gpu_lowpriority
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
PYTHON="${PYTHON:-/home/muqsitamir/miniconda3/envs/harmonit311/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/harmonized/diffusion_img2img_nyu_s20000_strength035}"
SITE_PROBE="${SITE_PROBE:-checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt}"

cd "${REPO_DIR}"
mkdir -p logs "${OUT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/methods/diffusion_img2img.py train \
  --out-dir "${OUT_DIR}" \
  --steps 20000 \
  --batch-size 16 \
  --num-workers 6 \
  --base-channels 64 \
  --channel-mults 1 2 4 4 \
  --num-timesteps 1000 \
  --log-every 100 \
  --save-every 2000

"${PYTHON}" scripts/methods/diffusion_img2img.py export \
  --checkpoint "${OUT_DIR}/checkpoints/model_latest.pt" \
  --out-dir "${OUT_DIR}" \
  --batch-size 8 \
  --num-workers 4 \
  --ddim-steps 50 \
  --strength 0.35

if [ -f scripts/eval_harmonized_npz.py ]; then
  "${PYTHON}" scripts/eval_harmonized_npz.py \
    --npz-path "${OUT_DIR}/test/diffusion_img2img_nyu_slices.npz" \
    --out-dir "${OUT_DIR}/test/metrics" \
    --site-probe-ckpt "${SITE_PROBE}" \
    --batch-size 16 \
    --num-workers 4
fi
