#!/bin/bash
# Protocol amendment 13: HACA3 (authors' pretrained model) on every ABIDE subject, exported on the
# frozen slices. Test subjects go first so the probe script can start; every step skips finished
# subjects, so the script can be rerun after an interruption.
# Usage: ssh -n vpulab '(setsid nohup bash .../scripts/vpulab_isbi2027_haca3.sh > .../haca3/run.log 2>&1 &)'
set -uo pipefail
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
DATA_REPO="${DATA_REPO:-/mnt/rhome/mmi/projects/HarmonIt}"
B="$ROOT/haca3"
OUT="$B/run"
PY="${HACA3_PYTHON:-/home/mmi/envs/haca3/bin/python}"
HPY="${HARMONIT_PYTHON:-/home/mmi/envs/harmonit-isbi/bin/python}"
export PYTHONPATH="$ISBI_CODE/src" PYTHONUNBUFFERED=1
S="$PY $ISBI_CODE/scripts/methods/haca3_abide.py"
C="--out-dir $OUT --manifest-path $DATA_REPO/data/abide_manifest.csv --splits-path $DATA_REPO/data/splits.json"
T="--template $B/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
W="--harmonization-model $B/weights/harmonization_public.pt --fusion-model $B/weights/fusion.pt"
log() { echo "$(date +%F_%T) $*"; }
sha256sum "$ISBI_CODE/scripts/methods/haca3_abide.py" "$B"/weights/*.pt "$B"/tpl-*.nii.gz
git -C "$B/haca3" rev-parse HEAD

log "prepare NYU training volumes"; $S prepare $C $T --split train --site NYU
[ -e "$OUT/target.json" ] || { log "select target"; $S target $C $W; }
log "prepare test"; $S prepare $C $T --split test
log "harmonize test"; $S harmonize $C $W
[ -e "$OUT/export/test/haca3_slices.npz" ] || { log "export test"
  $S export $C --split test --slice-index-map "$ISBI_CODE/configs/isbi2027/test_slice_indices.json"; }

log "prepare val and train (background) while harmonizing"
( $S prepare $C $T --split val --workers 7 && $S prepare $C $T --split train --workers 7 ) & prep=$!
while kill -0 "$prep" 2>/dev/null; do $S harmonize $C $W; sleep 60; done
wait "$prep" || { log "PREPARE FAILED"; exit 1; }
$S harmonize $C $W

E="$ROOT/exports/histogram_matching"
[ -e "$E/val" ] || { log "histogram-matching val export (val slice indices)"
  ( cd "$ROOT/probe_work" && "$HPY" "$ISBI_CODE/scripts/methods/histogram_matching.py" --max-reference-subjects 256 \
      --num-workers 2 --split val --out-dir "$E" > "$ROOT/exports/histogram_matching_val.log" 2>&1 ) \
    || { log "HISTOGRAM VAL EXPORT FAILED"; exit 1; }; }
for split in val train; do
  [ -e "$OUT/export/$split/haca3_slices.npz" ] || { log "export $split"
    $S export $C --split "$split" --slice-index-map "$E/$split/histogram_matching_slices.npz"; }
done
log "haca3 finished"
