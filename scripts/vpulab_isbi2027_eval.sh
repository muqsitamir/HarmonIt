#!/bin/bash
# ISBI 2027 corrected evaluation on vpulab (RTX A5000). Mirrors slurm/isbi2027_eval.sbatch.
# Usage: ISBI_OUTPUT=/path/to/new/run nohup bash scripts/vpulab_isbi2027_eval.sh > run.log 2>&1 &
set -euo pipefail
: "${ISBI_OUTPUT:?Set a new evaluation output directory}"
ISBI_CODE="${ISBI_CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
ISBI_INPUTS="${ISBI_INPUTS:-$DATA_REPO}"
# Canonical re-export (slurm/isbi2027_hcld_reexport.sbatch); the historical cl export
# tie-broke SBL_51570 to slice 68. HCLD_ARTIFACT= (empty) omits adapted_hcld.
HCLD="${HCLD_ARTIFACT-/mnt/rhome/mmi/projects/isbi2027/inputs/adapted_hcld_isbi2027_canonical/test/adapted_hcld_slices.npz}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
# Second sampling draw of diffusion 20k (vpulab re-export); added when present. REDRAW= omits it.
REDRAW_DEFAULT=/mnt/rhome/mmi/projects/isbi2027/exports/diffusion_20k/test/diffusion_img2img_nyu_slices.npz
[ -f "$REDRAW_DEFAULT" ] || REDRAW_DEFAULT=
REDRAW="${REDRAW-$REDRAW_DEFAULT}"
export PYTHONPATH="${ISBI_CODE}/src:${ISBI_CODE}/scripts"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ISBI_CODE"
"$PY" -m unittest discover -s tests -v
"$PY" scripts/eval_isbi2027.py \
  --manifest-path "$DATA_REPO/data/abide_manifest.csv" \
  --splits-path "$DATA_REPO/data/splits.json" \
  --site-probe-ckpt "${SITE_PROBE:-$ISBI_INPUTS/checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt}" \
  --slice-map "$ISBI_CODE/configs/isbi2027/test_slice_indices.json" \
  --out-dir "$ISBI_OUTPUT" --num-workers "${NUM_WORKERS:-4}" \
  --artifact "neurocombat=$ISBI_INPUTS/outputs/harmonized/neurocombat/test/neurocombat_slices.npz" \
  --artifact "histogram_matching=$ISBI_INPUTS/outputs/harmonized/histogram_matching/test/histogram_matching_slices.npz" \
  --artifact "cyclegan_tuned=$ISBI_INPUTS/outputs/harmonized/cyclegan_nyu_id1_s1500/test/cyclegan_nyu_slices.npz" \
  --artifact "stargan_aggressive=$ISBI_INPUTS/outputs/harmonized/stargan_nyu_aggr/test/stargan_nyu_slices.npz" \
  --artifact "stargan_conservative=$ISBI_INPUTS/outputs/harmonized/stargan_nyu/test/stargan_nyu_slices.npz" \
  --artifact "dlest_1500=$ISBI_INPUTS/outputs/harmonized/dlest_nyu/test/dlest_nyu_slices.npz" \
  --artifact "dlest_1000=$ISBI_INPUTS/outputs/harmonized/dlest_nyu_step1000/test/dlest_nyu_slices.npz" \
  --artifact "diffusion_20k=$ISBI_INPUTS/outputs/harmonized/diffusion_img2img_nyu_s20000_strength035/test/diffusion_img2img_nyu_slices.npz" \
  ${HCLD:+--artifact "adapted_hcld=$HCLD"} \
  ${REDRAW:+--artifact "diffusion_20k_redraw=$REDRAW"} \
  ${EXTRA_ARTIFACT:+--artifact "$EXTRA_ARTIFACT"} \
  ${PROBE_INPUT_MASK:+--probe-input-mask "$PROBE_INPUT_MASK"}
