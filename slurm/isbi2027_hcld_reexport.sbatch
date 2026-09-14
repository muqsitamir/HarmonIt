#!/bin/bash
#SBATCH --job-name=harmonit_isbi_hcld_reexport
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Re-export adapted HCLD test outputs with the frozen ISBI slice map
# (configs/isbi2027/test_slice_indices.json). Same checkpoints, seed and settings as
# slurm/hcld_adapted_export_eval.sbatch; only the slice map and output directory differ.
# The historical export tie-broke SBL_51570 to slice 68; the canonical map uses 69.
# Evaluation is done on vpulab against the fresh reference, not here.
set -euo pipefail
: "${SLICE_INDEX_MAP:?Set path to the frozen canonical slice map}"
REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
HCLD_PYTHON="${HCLD_PYTHON:-/home/muqsitamir/miniconda3/envs/hcld/bin/python}"
AE_CKPT="${AE_CKPT:-outputs/hcld_abide/adapted_hcld/aekl/checkpoints/model_latest.pt}"
LDM_CKPT="${LDM_CKPT:-outputs/hcld_abide/adapted_hcld/ldm/checkpoints/model_latest.pt}"
OUT_DIR="${OUT_DIR:-outputs/harmonized/adapted_hcld_isbi2027_canonical}"

cd "${REPO_DIR}"
if [ -e "${OUT_DIR}" ]; then
  echo "Refusing to overwrite existing ${OUT_DIR}" >&2
  exit 1
fi
mkdir -p logs
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
sha256sum "${SLICE_INDEX_MAP}" "${AE_CKPT}" "${LDM_CKPT}" scripts/methods/hcld_export_ldm.py src/harmonit/data/abide_slices_dataset.py
nvidia-smi --query-gpu=name --format=csv,noheader

"${HCLD_PYTHON}" scripts/methods/hcld_export_ldm.py \
  --hcld-root /home/muqsitamir/repos/HCLD \
  --data-dir outputs/hcld_abide \
  --ae-checkpoint "${AE_CKPT}" \
  --ldm-checkpoint "${LDM_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --split test \
  --target-site-id 5 \
  --condition-labels labels/train_tar.tsv \
  --valid-nonzero-frac 0.02 \
  --fg-bbox-thr 0.02 \
  --slice-index-map "${SLICE_INDEX_MAP}" \
  --num-workers 0 \
  --amp-dtype bf16
