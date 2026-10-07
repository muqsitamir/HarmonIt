#!/bin/bash
# Protocol amendments 15 (preprocessing-only control for HACA3) and 16 (five-fold cross-validation
# of HACA3 decodability over all subjects). Reuses amendment 13's registrations, HACA3 outputs and
# retrained probes; every step skips finished work.
# Usage: ssh -n vpulab '(setsid nohup bash .../scripts/vpulab_isbi2027_haca3_controls.sh > .../haca3/controls.log 2>&1 &)'
set -uo pipefail
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
HPY="${HACA3_PYTHON:-/home/mmi/envs/haca3/bin/python}"
B="$MAIN/haca3"; OUT="$B/run"; P="$B/probes"; CTRL="$B/run/export_preproc"
log() { echo "$(date +%F_%T) $*"; }
C="--out-dir $OUT --manifest-path $DATA_REPO/data/abide_manifest.csv --splits-path $DATA_REPO/data/splits.json"

# 1. Control exports (written to export/<split>/haca3_preproc_slices.npz, then moved apart).
export PYTHONPATH="$ISBI_CODE/src" PYTHONUNBUFFERED=1
for split in test val train; do
  [ -e "$CTRL/$split/haca3_preproc_slices.npz" ] && continue
  map=$([ "$split" = test ] && echo "$ISBI_CODE/configs/isbi2027/test_slice_indices.json" \
        || echo "$MAIN/exports/histogram_matching/$split/histogram_matching_slices.npz")
  log "export control $split"
  "$HPY" "$ISBI_CODE/scripts/methods/haca3_abide.py" export $C --source preproc --split "$split" --slice-index-map "$map" \
    || { log "CONTROL EXPORT FAILED $split"; exit 1; }
  mkdir -p "$CTRL/$split" && mv "$OUT/export/$split/haca3_preproc_slices.npz" "$CTRL/$split/"
done

# 2. Probes of amendment 13 on HACA3 and the control (paired in one run per probe).
export PYTHONPATH="$ISBI_CODE/src:$ISBI_CODE/scripts" OMP_NUM_THREADS=4
evaluate() {  # name checkpoint
  local run="$P/runs_ctrl/$1"; mkdir -p "$P/runs_ctrl"
  [ -e "$run/COMPLETE.json" ] && return 0
  "$PY" "$ISBI_CODE/scripts/eval_isbi2027.py" --manifest-path "$DATA_REPO/data/abide_manifest.csv" \
    --splits-path "$DATA_REPO/data/splits.json" --site-probe-ckpt "$2" \
    --slice-map "$ISBI_CODE/configs/isbi2027/test_slice_indices.json" --out-dir "$run" --num-workers 4 \
    --artifact "haca3=$OUT/export/test/haca3_slices.npz" --artifact "haca3_preproc=$CTRL/test/haca3_preproc_slices.npz" \
    > "$run.log" 2>&1 && log "evaluated $1" || log "EVAL FAILED $1"
}
evaluate frozen_probe_v1_9methods_haca3ctrl "$DATA_REPO/checkpoints/site_probe_v0.3_aug_ramp15/model_best.pt"
for seed in 42 1 2 3 4; do
  evaluate "retrained_raw_seed${seed}_model_best_9methods_haca3ctrl" \
    "$(ls "$P"/probe_work/runs/site_probe/isbi2027__raw_seed${seed}/*/model_best.pt | tail -1)"
done
for seed in 5 6 7 8 9; do
  evaluate "converged_raw_seed${seed}_model_last_9methods_haca3ctrl" \
    "$(ls "$P"/probe_work/runs/site_probe/isbi2027__converged_raw_seed${seed}/*/model_last.pt | tail -1)"
done
for seed in 1 2 3; do
  evaluate "sliceprobe_raw_seed${seed}_model_best_haca3ctrl" "$P/slice_probes/raw_seed${seed}/model_best.pt"
done

# 3. Intensity-only probes (amendment 14 script) on HACA3 and the control.
"$PY" "$ISBI_CODE/scripts/isbi2027_histogram_probe.py" --exports "$MAIN/exports" \
  --eval-run "$P/runs_ctrl/frozen_probe_v1_9methods_haca3ctrl" --methods haca3 haca3_preproc \
  --export "haca3=$OUT/export" --export "haca3_preproc=$CTRL" \
  --out "$P/analysis/histogram_probe_haca3ctrl.json" && log "histogram probes done" || log "HISTOGRAM FAILED"

# 4. Cross-validation (amendment 16).
"$PY" "$ISBI_CODE/scripts/isbi2027_haca3_cv.py" --haca3-export "$OUT/export" --preproc-export "$CTRL" \
  --out "$P/analysis/haca3_cv.json" && log "cross-validation done" || log "CV FAILED"
log "controls finished"
