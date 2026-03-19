#!/usr/bin/env bash
set -euo pipefail

docker run --rm -it \
  -v "$(pwd)":/workspace/HarmonIt \
  -w /workspace/HarmonIt \
  -e PYTHONPATH=/workspace/HarmonIt \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  harmonit-cpu:latest "$@"
