#!/bin/bash
#SBATCH --job-name=harmonit_hcld_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
PYTHON="${PYTHON:-/home/muqsitamir/miniconda3/envs/harmonit311/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/hcld_abide}"

cd "${REPO_DIR}"
mkdir -p logs "${OUT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/methods/hcld_prepare_abide.py \
  --out-dir "${OUT_DIR}" \
  --out-shape 192 192 64 \
  --target-site-id 5
