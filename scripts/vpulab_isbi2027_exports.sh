#!/bin/bash
# Export fixed-slice train/val (and a test reproduction) for the protocol's pre-selected
# harmonized-probe methods: histogram matching, tuned CycleGAN, diffusion img2img 20k.
# Settings match the historical test artifacts. Outputs go to $ROOT/exports/<method>/<split>.
set -uo pipefail
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
SPLITS="${SPLITS:-test val train}"
export PYTHONPATH="$ISBI_CODE/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
mkdir -p "$ROOT/probe_work" "$ROOT/exports"
[ -e "$ROOT/probe_work/data" ] || ln -s "$DATA_REPO/data" "$ROOT/probe_work/data"
cd "$ROOT/probe_work"  # method scripts resolve data/ relative to cwd

run() {  # name split command...
  local name="$1" split="$2"; shift 2
  if [ -e "$ROOT/exports/$name/$split" ]; then echo "skip existing $name/$split"; return; fi
  echo "$(date +%F_%T) export $name $split"
  "$@" --split "$split" --out-dir "$ROOT/exports/$name" > "$ROOT/exports/${name}_${split}.log" 2>&1 \
    && echo "$(date +%F_%T) done $name $split" || echo "EXPORT FAILED $name $split"
}

for split in $SPLITS; do
  run histogram_matching "$split" "$PY" "$ISBI_CODE/scripts/methods/histogram_matching.py" \
    --max-reference-subjects 256 --num-workers 2
  run cyclegan_tuned "$split" "$PY" "$ISBI_CODE/scripts/methods/cyclegan_many_to_one.py" export \
    --checkpoint "$DATA_REPO/outputs/harmonized/cyclegan_nyu_id1_s1500/train/checkpoints/latest.pt" --num-workers 2
  run diffusion_20k "$split" "$PY" "$ISBI_CODE/scripts/methods/diffusion_img2img.py" export \
    --checkpoint "$ROOT/inputs/diffusion_20k/checkpoints/model_latest.pt" --batch-size 8 --num-workers 2 \
    --ddim-steps 50 --strength 0.35
done
echo "$(date +%F_%T) exports finished"
