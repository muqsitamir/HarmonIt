#!/bin/bash
# Harmonized-probe experiment (protocol amendment 2): wait for exports, gate on the
# integrity check, then train slice probes per source and evaluate each checkpoint once.
set -uo pipefail
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
H="$DATA_REPO/outputs/harmonized"
E="$ROOT/exports"
SEEDS="${SEEDS:-1 2 3}"
export PYTHONPATH="$ISBI_CODE/src:$ISBI_CODE/scripts" PYTHONUNBUFFERED=1

until grep -q "exports finished" "$ROOT/exports_run.log" 2>/dev/null; do sleep 120; done
grep -q "EXPORT FAILED" "$ROOT/exports_run.log" && { echo "exports failed; stopping"; exit 1; }
"$PY" "$ISBI_CODE/scripts/check_isbi2027_exports.py" --exports "$E" --splits-path "$DATA_REPO/data/splits.json" \
  --historical "histogram_matching=$H/histogram_matching/test/histogram_matching_slices.npz" \
  --historical "cyclegan_tuned=$H/cyclegan_nyu_id1_s1500/test/cyclegan_nyu_slices.npz" \
  --historical "diffusion_20k=$H/diffusion_img2img_nyu_s20000_strength035/test/diffusion_img2img_nyu_slices.npz" \
  --report "$E/integrity_report.json" || { echo "integrity check failed; stopping"; exit 1; }

declare -A NPZ=([histogram_matching]=histogram_matching_slices.npz [cyclegan_tuned]=cyclegan_nyu_slices.npz \
  [diffusion_20k]=diffusion_img2img_nyu_slices.npz)
PROBES="$ROOT/slice_probes"
mkdir -p "$PROBES"

train() {  # tag method image_key seed [extra]
  local tag="$1" method="$2" key="$3" seed="$4"; shift 4
  local out="$PROBES/${tag}_seed${seed}"
  [ -e "$out/model_last.pt" ] && { echo "skip $out"; return 0; }
  echo "$(date +%F_%T) train $tag seed $seed"
  "$PY" "$ISBI_CODE/scripts/train_slice_probe.py" --seed "$seed" --image-key "$key" \
    --train-npz "$E/$method/train/${NPZ[$method]}" --val-npz "$E/$method/val/${NPZ[$method]}" \
    --splits-path "$DATA_REPO/data/splits.json" --out-dir "$out" "$@" > "$out.log" 2>&1 \
    || { echo "TRAIN FAILED $out"; return 1; }
  for ckpt in model_best model_last; do
    local run="$ROOT/runs/sliceprobe_${tag}_seed${seed}_${ckpt}_9methods"
    [ -e "$run/COMPLETE.json" ] && continue
    SITE_PROBE="$out/$ckpt.pt" ISBI_OUTPUT="$run" bash "$ISBI_CODE/scripts/vpulab_isbi2027_eval.sh" > "$run.log" 2>&1 \
      && echo "evaluated $run" || echo "EVAL FAILED $run"
  done
}

train raw_shuffle histogram_matching raw_images 1 --label-shuffle
for seed in $SEEDS; do
  train raw histogram_matching raw_images "$seed"
  for method in histogram_matching cyclegan_tuned diffusion_20k; do
    train "$method" "$method" images "$seed"
  done
done
echo "$(date +%F_%T) slice probes finished"
