#!/bin/bash
# ISBI 2027 site-probe retraining on vpulab with the production v0.3_aug_ramp15 recipe
# (checkpoint run 20260417_164204). LABEL_SHUFFLE=1 trains the subject-level shuffle control.
# Usage: SEED=42 LABEL_SHUFFLE=0 nohup bash scripts/vpulab_isbi2027_probe.sh > probe.log 2>&1 &
set -euo pipefail
: "${SEED:?Set SEED}"
: "${LABEL_SHUFFLE:?Set LABEL_SHUFFLE=0 (raw) or 1 (shuffle control)}"
ISBI_CODE="${ISBI_CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
WORK="${PROBE_WORK:-/mnt/rhome/mmi/projects/isbi2027/probe_work}"
PY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
KIND=$([ "$LABEL_SHUFFLE" = 1 ] && echo shuffle || echo raw)

mkdir -p "$WORK" "$HOME/mlflow_local"
[ -e "$WORK/data" ] || ln -s "$DATA_REPO/data" "$WORK/data"
cd "$WORK"  # train_site_probe.py resolves data/ and runs/ relative to cwd

export PYTHONPATH="$ISBI_CODE/src"
export ABLATION_NAME="isbi2027__${KIND}_seed${SEED}"
export MLFLOW_TRACKING_URI="sqlite:///$HOME/mlflow_local/isbi2027.db"
export MLFLOW_ARTIFACT_ROOT="file://$WORK/mlruns"
export BATCH_SIZE=64 EPOCHS=10 LR=3e-4 STEPS_PER_EPOCH=50 VAL_BATCHES=0
export MASK_MODE=none BG_SUPPRESS=1 INPUT_MODE=image MASK_ONLY_REPR=binary
export FG_THR=0.02 FG_BBOX_THR=0.02 HEAD_MASK_THR=0.02 HEAD_MASK_DILATE=3
export AUG_AFFINE=1 AUG_PROB=0.9 AUG_ROT_DEG=12 AUG_TRANS_PX=32 AUG_SCALE_JITTER=0.2

echo "code_commit=$(cat "$ISBI_CODE/COMMIT" 2>/dev/null || echo unknown)"
sha256sum "$ISBI_CODE/scripts/train_site_probe.py" "$ISBI_CODE/src/harmonit/data/abide_slices_dataset.py" \
  "$ISBI_CODE/src/harmonit/data/label_controls.py"
exec "$PY" "$ISBI_CODE/scripts/train_site_probe.py"
