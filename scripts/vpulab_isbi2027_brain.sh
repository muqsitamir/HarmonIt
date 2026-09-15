#!/bin/bash
# Brain-only control (protocol amendment 7): wait for HD-BET masks, build brain-only exports
# (gated on exact raw-slice reproduction), train slice probes and evaluate each checkpoint once
# with brain-only probe inputs, then run the brain-foreground intensity-histogram probe.
# Usage: setsid nohup bash scripts/vpulab_isbi2027_brain.sh > brain_run.log 2>&1 < /dev/null &
set -uo pipefail
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
E="$ROOT/exports"
B="$E/brain"
SEEDS="${SEEDS:-1 2 3}"
export PYTHONPATH="$ISBI_CODE/src:$ISBI_CODE/scripts" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2

until grep -q "^finished; missing masks" "$ROOT/probe_logs/hdbet_masks.log" 2>/dev/null; do sleep 120; done
grep -q "^finished; missing masks: \[\]" "$ROOT/probe_logs/hdbet_masks.log" || { echo "HD-BET masks missing; stopping"; exit 1; }
[ -f "$B/brain_mask_report.json" ] || "$PY" "$ISBI_CODE/scripts/make_brain_npz.py" --exports "$E" \
  --mask-dir "$ROOT/brain_masks/hdbet" --manifest "$DATA_REPO/data/abide_manifest.csv" \
  --splits "$DATA_REPO/data/splits.json" --out-dir "$B" --volume-cache "${VOLUME_CACHE_DIR:-/home/mmi/cache/isbi2027_volumes}" \
  || { echo "brain export gate failed; stopping"; exit 1; }

declare -A NPZ=([histogram_matching]=histogram_matching_slices.npz [cyclegan_tuned]=cyclegan_nyu_slices.npz \
  [diffusion_20k]=diffusion_img2img_nyu_slices.npz [brain_shape]=brain_shape_slices.npz)
PROBES="$ROOT/brain_probes"
mkdir -p "$PROBES"

train() {  # tag export image_key seed [extra]
  local tag="$1" method="$2" key="$3" seed="$4"; shift 4
  local out="$PROBES/${tag}_seed${seed}"
  [ -e "$out/model_last.pt" ] || { echo "$(date +%F_%T) train $tag seed $seed"
    "$PY" "$ISBI_CODE/scripts/train_slice_probe.py" --seed "$seed" --image-key "$key" \
      --train-npz "$B/$method/train/${NPZ[$method]}" --val-npz "$B/$method/val/${NPZ[$method]}" \
      --splits-path "$DATA_REPO/data/splits.json" --out-dir "$out" "$@" > "$out.log" 2>&1 \
      || { echo "TRAIN FAILED $out"; return 1; }; }
  for ckpt in model_best model_last; do
    local run="$ROOT/runs/brainprobe_${tag}_seed${seed}_${ckpt}"
    [ -e "$run/COMPLETE.json" ] && continue
    SITE_PROBE="$out/$ckpt.pt" PROBE_INPUT_MASK="$B/masks/test/brain_masks.npz" ISBI_OUTPUT="$run" \
      EXTRA_ARTIFACT="brain_shape=$B/brain_shape/test/brain_shape_slices.npz" \
      bash "$ISBI_CODE/scripts/vpulab_isbi2027_eval.sh" > "$run.log" 2>&1 && echo "evaluated $run" || echo "EVAL FAILED $run"
  done
}

train raw_shuffle histogram_matching raw_images 1 --label-shuffle
for seed in $SEEDS; do
  train raw histogram_matching raw_images "$seed"
  for method in histogram_matching cyclegan_tuned diffusion_20k brain_shape; do
    train "$method" "$method" images "$seed"
  done
done

run=$(ls -d "$ROOT"/runs/brainprobe_raw_seed1_model_best 2>/dev/null)
[ -f "$ROOT/analysis/histogram_probe_brain.json" ] || "$PY" "$ISBI_CODE/scripts/isbi2027_histogram_probe.py" \
  --exports "$E" --eval-run "$run" --brain-masks "$B/masks" --out "$ROOT/analysis/histogram_probe_brain.json"
echo "$(date +%F_%T) brain-only control finished"
