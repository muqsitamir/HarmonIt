#!/bin/bash
# Protocol amendment 17: brain-only probes for HACA3 (amendment 7 applied to amendment 13's outputs).
# Usage: ssh -n vpulab '(setsid nohup bash .../scripts/vpulab_isbi2027_haca3_brain.sh > .../haca3/brain.log 2>&1 &)'
set -uo pipefail
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
BETPY="${HDBET_PYTHON:-/home/mmi/envs/hdbet/bin/python}"
OUT="$MAIN/haca3/run"; P="$MAIN/haca3/probes"; MASKS="$MAIN/brain_masks/hdbet"; BR="$MAIN/haca3/brain"
log() { echo "$(date +%F_%T) $*"; }

log "HD-BET masks"
"$BETPY" "$ISBI_CODE/scripts/run_hdbet_masks.py" --data-repo "$DATA_REPO" --out-dir "$MASKS" \
  || { log "HD-BET FAILED"; exit 1; }
missing=$(comm -23 <(tail -n +2 "$DATA_REPO/data/abide_manifest.csv" | cut -d, -f1 | sort) <(ls "$MASKS" | sed 's/.nii.gz//' | sort) | wc -l)
[ "$missing" = 0 ] || { log "MISSING MASKS: $missing"; exit 1; }

export PYTHONPATH="$ISBI_CODE/src:$ISBI_CODE/scripts" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
[ -f "$BR/brain_mask_report.json" ] || { log "brain exports"; "$PY" "$ISBI_CODE/scripts/make_brain_npz.py" \
  --exports "$MAIN/exports" --mask-dir "$MASKS" --manifest "$DATA_REPO/data/abide_manifest.csv" \
  --splits "$DATA_REPO/data/splits.json" --out-dir "$BR" --volume-cache /home/mmi/cache/isbi2027_volumes \
  --methods haca3 --export "haca3=$OUT/export" --reference haca3 || { log "BRAIN EXPORT GATE FAILED"; exit 1; }; }

mkdir -p "$P/brain_probes" "$P/runs_brain"
train() {  # tag image_key seed
  local tag="$1" key="$2" seed="$3" out="$P/brain_probes/$1_seed$3"
  [ -e "$out/model_last.pt" ] || { log "brain probe $tag seed $seed"
    "$PY" "$ISBI_CODE/scripts/train_slice_probe.py" --seed "$seed" --image-key "$key" \
      --train-npz "$BR/haca3/train/haca3_slices.npz" --val-npz "$BR/haca3/val/haca3_slices.npz" \
      --splits-path "$DATA_REPO/data/splits.json" --out-dir "$out" > "$out.log" 2>&1 || { log "TRAIN FAILED $out"; return 1; }; }
  for ckpt in model_best model_last; do
    local run="$P/runs_brain/brainprobe_${tag}_seed${seed}_${ckpt}"
    [ -e "$run/COMPLETE.json" ] && continue
    "$PY" "$ISBI_CODE/scripts/eval_isbi2027.py" --manifest-path "$DATA_REPO/data/abide_manifest.csv" \
      --splits-path "$DATA_REPO/data/splits.json" --site-probe-ckpt "$out/$ckpt.pt" \
      --slice-map "$ISBI_CODE/configs/isbi2027/test_slice_indices.json" --out-dir "$run" --num-workers 4 \
      --artifact "haca3=$OUT/export/test/haca3_slices.npz" --probe-input-mask "$BR/masks/test/brain_masks.npz" \
      > "$run.log" 2>&1 && log "evaluated $run" || log "EVAL FAILED $run"
  done
}
for seed in 1 2 3; do train raw raw_images "$seed"; train haca3 images "$seed"; done

for ckpt in best last; do
  "$PY" "$ISBI_CODE/scripts/isbi2027_probe_difference.py" --haca3 --runs "$P/runs_brain" --ckpt "$ckpt" \
    --out "$P/analysis/probe_difference_haca3_brain_$ckpt.json" || log "DIFFERENCE FAILED $ckpt"
done
"$PY" "$ISBI_CODE/scripts/isbi2027_histogram_probe.py" --exports "$MAIN/exports" \
  --eval-run "$P/runs_brain/brainprobe_raw_seed1_model_best" --brain-masks "$BR/masks" --methods haca3 \
  --export "haca3=$OUT/export" --out "$P/analysis/histogram_probe_haca3_brain.json" || log "HISTOGRAM FAILED"
log "brain-only HACA3 finished"
