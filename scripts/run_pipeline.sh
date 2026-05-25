#!/usr/bin/env bash
# Run the whole DRC pipeline end to end on a single machine.
# Stops at the first failing stage. Override the config by exporting CONFIG.
set -euo pipefail

CONFIG="${CONFIG:-configs/base.yaml}"
PY="${PYTHON:-python}"

echo ">> Data: download, parse, audit, dose corpora"
$PY -m drc.data.download     --config "$CONFIG"
$PY -m drc.data.parse        --config "$CONFIG"
$PY -m drc.data.qa_audit     --config "$CONFIG" --judge manual
$PY -m drc.data.dose_corpora --config "$CONFIG"

echo ">> Tokenizer"
$PY -m drc.tokenizer.train_tokenizer --config "$CONFIG"

echo ">> Training sweep (pilot-gated)"
$PY -m drc.train.sweep --config "$CONFIG"

echo ">> Evaluation"
$PY -m drc.eval.run_eval       --config "$CONFIG"
$PY -m drc.eval.ngram_baseline --config "$CONFIG"

echo ">> Analysis + figures"
$PY -m drc.analysis.hill             --config "$CONFIG"
$PY -m drc.analysis.model_comparison --config "$CONFIG"
$PY -m drc.analysis.clustering       --config "$CONFIG"
$PY -m drc.analysis.decision         --config "$CONFIG"
$PY -m drc.analysis.figures          --config "$CONFIG"

echo ">> Done. See results/ and results/figures/."
