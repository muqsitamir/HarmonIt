#!/bin/bash
# Protocol amendment 13 probes. (b) Retrain the benchmark-recipe (seeds 42 1 2 3 4) and converged
# (seeds 5-9) raw site probes with the original recipes and seeds, and evaluate each on the eight
# test artifacts plus HACA3. (c) Whole-head slice probes trained on raw slices
# and on HACA3 outputs (seeds 1-3). Everything goes under haca3/probes, apart from the paper's runs.
# Usage: ssh -n vpulab '(setsid nohup bash .../scripts/vpulab_isbi2027_haca3_probes.sh > .../haca3/probes.log 2>&1 &)'
set -uo pipefail
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
OUT="$MAIN/haca3/run"
P="$MAIN/haca3/probes"
export ISBI_ROOT="$P" PROBE_WORK="$P/probe_work"
export EXTRA_ARTIFACT="haca3=$OUT/export/test/haca3_slices.npz" HCLD_ARTIFACT= REDRAW=
mkdir -p "$P/runs" "$P/slice_probes"
log() { echo "$(date +%F_%T) $*"; }

until [ -e "$OUT/export/test/haca3_slices.npz" ]; do sleep 120; done
log "benchmark-recipe probes"
SEEDS="42 1 2 3 4" bash "$ISBI_CODE/scripts/vpulab_isbi2027_probe_seeds.sh"
log "converged probes"
RECIPE=converged VOLUME_CACHE_DIR="${VOLUME_CACHE_DIR:-/home/mmi/cache/isbi2027_volumes}" SEEDS="5 6 7 8 9" \
  bash "$ISBI_CODE/scripts/vpulab_isbi2027_probe_seeds.sh"

until grep -q "haca3 finished" "$MAIN/haca3/run.log" 2>/dev/null; do sleep 300; done
export PYTHONPATH="$ISBI_CODE/src:$ISBI_CODE/scripts" PYTHONUNBUFFERED=1
train() {  # tag image_key seed
  local tag="$1" key="$2" seed="$3" out="$P/slice_probes/$1_seed$3"
  if [ ! -e "$out/model_last.pt" ]; then
    log "slice probe $tag seed $seed"
    "$PY" "$ISBI_CODE/scripts/train_slice_probe.py" --seed "$seed" --image-key "$key" \
      --train-npz "$OUT/export/train/haca3_slices.npz" --val-npz "$OUT/export/val/haca3_slices.npz" \
      --splits-path "$DATA_REPO/data/splits.json" --out-dir "$out" > "$out.log" 2>&1 || { log "TRAIN FAILED $out"; return 1; }
  fi
  for ckpt in model_best model_last; do
    local run="$P/runs/sliceprobe_${tag}_seed${seed}_${ckpt}_haca3"
    [ -e "$run/COMPLETE.json" ] && continue
    SITE_PROBE="$out/$ckpt.pt" ISBI_OUTPUT="$run" bash "$ISBI_CODE/scripts/vpulab_isbi2027_eval.sh" > "$run.log" 2>&1 \
      && log "evaluated $run" || log "EVAL FAILED $run"
  done
}
for seed in 1 2 3; do
  train raw raw_images "$seed"
  train haca3 images "$seed"
done
log "haca3 probes finished"
