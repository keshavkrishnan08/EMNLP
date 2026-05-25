#!/usr/bin/env bash
# Launch the 60-run sweep on a Kaggle dual-T4 session.
# The sweep manager pins runs to GPUs via CUDA_VISIBLE_DEVICES and keeps two
# runs in flight. It runs the pilot (aann/all/42) first and refuses to continue
# if the pilot's held-out perplexity is out of band.
set -euo pipefail

CONFIG="${CONFIG:-configs/base.yaml}"

nvidia-smi || { echo "No GPU visible — this launcher needs CUDA."; exit 1; }

python -m drc.train.sweep --config "$CONFIG"
