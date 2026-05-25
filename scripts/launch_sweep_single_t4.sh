#!/usr/bin/env bash
# Single-T4 fallback. Runs sequentially and drops the dose=4 condition to fit
# 48 runs in a session (per the PRD graceful-degradation plan). Resume-safe:
# completed runs are skipped on restart.
set -euo pipefail

CONFIG="${CONFIG:-configs/base.yaml}"

nvidia-smi || { echo "No GPU visible — this launcher needs CUDA."; exit 1; }

python -m drc.train.sweep --config "$CONFIG" --single-gpu
