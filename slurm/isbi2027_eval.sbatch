#!/bin/bash
#SBATCH --job-name=harmonit_isbi_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${ISBI_CODE:?Set isolated code snapshot directory}"
: "${ISBI_INPUTS:?Set vpulab artifact snapshot directory}"
: "${ISBI_OUTPUT:?Set a new evaluation output directory}"
DATA_REPO="${DATA_REPO:-/home/muqsitamir/repos/HarmonIt}"
PY="${HARMONIT_PYTHON:-/home/muqsitamir/miniconda3/envs/harmonit311/bin/python}"
export PYTHONPATH="${ISBI_CODE}/src"
export LD_LIBRARY_PATH="/home/muqsitamir/miniconda3/envs/harmonit311/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ISBI_CODE"
"$PY" -m unittest discover -s tests -v
"$PY" scripts/eval_isbi2027.py \
  --manifest-path "$DATA_REPO/data/abide_manifest.csv" \
  --splits-path "$DATA_REPO/data/splits.json" \
  --site-probe-ckpt "$ISBI_INPUTS/checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt" \
  --out-dir "$ISBI_OUTPUT" --num-workers 2 \
  --artifact "neurocombat=$ISBI_INPUTS/outputs/harmonized/neurocombat/test/neurocombat_slices.npz" \
  --artifact "histogram_matching=$ISBI_INPUTS/outputs/harmonized/histogram_matching/test/histogram_matching_slices.npz" \
  --artifact "cyclegan_tuned=$ISBI_INPUTS/outputs/harmonized/cyclegan_nyu_id1_s1500/test/cyclegan_nyu_slices.npz" \
  --artifact "stargan_aggressive=$ISBI_INPUTS/outputs/harmonized/stargan_nyu_aggr/test/stargan_nyu_slices.npz" \
  --artifact "stargan_conservative=$ISBI_INPUTS/outputs/harmonized/stargan_nyu/test/stargan_nyu_slices.npz" \
  --artifact "dlest_1500=$ISBI_INPUTS/outputs/harmonized/dlest_nyu/test/dlest_nyu_slices.npz" \
  --artifact "dlest_1000=$ISBI_INPUTS/outputs/harmonized/dlest_nyu_step1000/test/dlest_nyu_slices.npz" \
  --artifact "diffusion_20k=$ISBI_INPUTS/outputs/harmonized/diffusion_img2img_nyu_s20000_strength035/test/diffusion_img2img_nyu_slices.npz" \
  --artifact "adapted_hcld=$DATA_REPO/outputs/harmonized/adapted_hcld/test/adapted_hcld_slices.npz"
