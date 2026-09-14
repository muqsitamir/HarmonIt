#!/bin/bash
# Silhouette control (protocol amendment 4): slice probes on filled binary head silhouettes.
set -uo pipefail
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
S="$ROOT/exports/silhouette"
export PYTHONPATH="$ISBI_CODE/src:$ISBI_CODE/scripts" PYTHONUNBUFFERED=1

for split in train val test; do
  [ -f "$S/$split/silhouette_slices.npz" ] || "$PY" "$ISBI_CODE/scripts/make_silhouette_npz.py" \
    --npz "$ROOT/exports/histogram_matching/$split/histogram_matching_slices.npz" --out "$S/$split/silhouette_slices.npz"
done
for seed in ${SEEDS:-1 2 3}; do
  out="$ROOT/slice_probes/silhouette_seed$seed"
  [ -e "$out/model_last.pt" ] || "$PY" "$ISBI_CODE/scripts/train_slice_probe.py" --seed "$seed" --image-key images \
    --train-npz "$S/train/silhouette_slices.npz" --val-npz "$S/val/silhouette_slices.npz" \
    --splits-path "$DATA_REPO/data/splits.json" --out-dir "$out" > "$out.log" 2>&1 || { echo "TRAIN FAILED $out"; continue; }
  for ckpt in model_best model_last; do
    run="$ROOT/runs/sliceprobe_silhouette_seed${seed}_${ckpt}_9methods"
    [ -e "$run/COMPLETE.json" ] && continue
    SITE_PROBE="$out/$ckpt.pt" EXTRA_ARTIFACT="silhouette=$S/test/silhouette_slices.npz" ISBI_OUTPUT="$run" \
      bash "$ISBI_CODE/scripts/vpulab_isbi2027_eval.sh" > "$run.log" 2>&1 && echo "evaluated $run" || echo "EVAL FAILED $run"
  done
done
echo "$(date +%F_%T) silhouette control finished"
