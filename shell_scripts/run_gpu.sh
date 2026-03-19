#!/usr/bin/env bash
set -euo pipefail

docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace/HarmonIt \
  -w /workspace/HarmonIt \
  -e PYTHONPATH=/workspace/HarmonIt \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  harmonit-gpu:latest "$@"
