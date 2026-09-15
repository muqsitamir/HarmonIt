#!/bin/bash
# Train raw site probes for several seeds (two concurrently on vpulab), then evaluate each
# selected checkpoint once on the nine frozen test artifacts.
# Usage: SEEDS="1 2 3 4" setsid nohup bash scripts/vpulab_isbi2027_probe_seeds.sh > seeds.log 2>&1 < /dev/null &
set -uo pipefail
SEEDS="${SEEDS:?Set SEEDS, e.g. \"1 2 3 4\"}"
KIND_SHUFFLE="${LABEL_SHUFFLE:-0}"
ISBI_CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ISBI_ROOT:-/mnt/rhome/mmi/projects/isbi2027}"
KIND=$([ "$KIND_SHUFFLE" = 1 ] && echo shuffle || echo raw)
RECIPE="${RECIPE:-production}"  # converged: protocol amendment 6
PREFIX=$([ "$RECIPE" = converged ] && echo converged || echo retrained)
TAG=$([ "$RECIPE" = converged ] && echo "converged_${KIND}" || echo "$KIND")
mkdir -p "$ROOT/probe_logs" "$ROOT/runs"

train() {
  SEED="$1" LABEL_SHUFFLE="$KIND_SHUFFLE" RECIPE="$RECIPE" bash "$ISBI_CODE/scripts/vpulab_isbi2027_probe.sh" \
    > "$ROOT/probe_logs/${TAG}_seed$1.log" 2>&1
}

evaluate() {
  local seed="$1" ckpt
  for ckpt in model_best model_last; do
    local pt run
    pt=$(ls "$ROOT"/probe_work/runs/site_probe/isbi2027__${TAG}_seed${seed}/*/${ckpt}.pt 2>/dev/null | tail -1)
    [ -n "$pt" ] || { echo "missing $ckpt for seed $seed"; continue; }
    run="$ROOT/runs/${PREFIX}_${KIND}_seed${seed}_${ckpt}_9methods_$(date +%Y%m%d_%H%M%S)"
    SITE_PROBE="$pt" ISBI_OUTPUT="$run" bash "$ISBI_CODE/scripts/vpulab_isbi2027_eval.sh" > "$run.log" 2>&1 \
      && echo "evaluated seed $seed $ckpt: $run" || echo "EVAL FAILED seed $seed $ckpt: $run.log"
  done
}

set -- $SEEDS
while [ $# -gt 0 ]; do
  a="$1"; shift; b="${1:-}"; [ -n "$b" ] && shift
  echo "$(date +%F_%T) training seeds $a ${b}"
  train "$a" & pa=$!
  [ -n "$b" ] && { sleep 90; train "$b" & pb=$!; }
  wait $pa && echo "seed $a done" || echo "SEED $a FAILED"
  [ -n "$b" ] && { wait $pb && echo "seed $b done" || echo "SEED $b FAILED"; }
  evaluate "$a"; [ -n "$b" ] && evaluate "$b"
done
echo "$(date +%F_%T) all seeds finished"
