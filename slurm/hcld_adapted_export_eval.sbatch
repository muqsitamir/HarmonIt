#!/bin/bash
#SBATCH --job-name=harmonit_adapt_hcld_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/muqsitamir/repos/HarmonIt}"
HCLD_PYTHON="${HCLD_PYTHON:-/home/muqsitamir/miniconda3/envs/hcld/bin/python}"
HARMONIT_PYTHON="${HARMONIT_PYTHON:-/home/muqsitamir/miniconda3/envs/harmonit311/bin/python}"
AE_CKPT="${AE_CKPT:-outputs/hcld_abide/adapted_hcld/aekl/checkpoints/model_latest.pt}"
LDM_CKPT="${LDM_CKPT:-outputs/hcld_abide/adapted_hcld/ldm/checkpoints/model_latest.pt}"
OUT_DIR="${OUT_DIR:-outputs/harmonized/adapted_hcld}"
SITE_PROBE="${SITE_PROBE:-checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt}"
SLICE_INDEX_MAP="${SLICE_INDEX_MAP:-${OUT_DIR}/test/eval_slice_indices.json}"

cd "${REPO_DIR}"
mkdir -p logs "${OUT_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

mkdir -p "$(dirname "${SLICE_INDEX_MAP}")"
"${HARMONIT_PYTHON}" - <<PY
import json
from pathlib import Path
from harmonit.data.abide_slices_dataset import AbideSlicesDataset

dataset = AbideSlicesDataset(
    manifest_path="data/abide_manifest.csv",
    splits_path="data/splits.json",
    split="test",
    out_hw=(256, 256),
    slice_mode="fixed",
    valid_nonzero_frac=0.02,
    fg_bbox_thr=0.02,
    seed=42,
    volume_cache_size=12,
    mask_mode="none",
    bg_suppress=True,
    input_mode="image",
)
dataset.aug_affine = False
slice_indices = {}
for i in range(len(dataset)):
    item = dataset[i]
    slice_indices[str(item[2])] = int(item[3])
Path("${SLICE_INDEX_MAP}").write_text(json.dumps(slice_indices, indent=2, sort_keys=True) + "\n")
PY

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

LD_LIBRARY_PATH="$("${HARMONIT_PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LIBDIR") or "")
PY
):${LD_LIBRARY_PATH:-}" \
"${HARMONIT_PYTHON}" scripts/eval_harmonized_npz.py \
  --npz-path "${OUT_DIR}/test/adapted_hcld_slices.npz" \
  --split test \
  --valid-nonzero-frac 0.02 \
  --fg-bbox-thr 0.02 \
  --site-probe-ckpt "${SITE_PROBE}" \
  --out-dir "${OUT_DIR}/test/metrics"
