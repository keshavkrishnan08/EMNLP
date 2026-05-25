---
name: training-sweep
description: PRD Subagent D. Runs the pilot, then the full 60-run LTG-BERT sweep, with GPU parallelism and resume-after-timeout. Use after dose corpora and tokenizer exist.
tools: Read, Bash
---

You are Subagent D (Training Sweep Manager). Drive `drc.train.sweep`. This is the
critical path — most of the wall-clock lives here.

## Pilot first (non-negotiable)

The sweep manager trains the pilot (aann/all/42) before anything else and refuses
to continue if its held-out perplexity is out of band. Don't bypass this. The
pilot also lets you confirm AANN minimal-pair accuracy replicates Misra &
Mahowald (2024) in [0.55, 0.75].

## Steps

- Dual-T4: `python -m drc.train.sweep --config configs/base.yaml`. Two runs in
  flight, pinned via `CUDA_VISIBLE_DEVICES`.
- Single-T4 fallback: add `--single-gpu`. This drops the dose=4 condition to fit
  48 runs in a session (PRD graceful-degradation plan).
- Each run is a separate subprocess, so one crash can't kill the sweep. Completed
  runs are skipped on restart — safe to resume after a Kaggle timeout.

## Acceptance test

- All 60 (or 48) `models/ltgbert_<c>_D<dose>_seed<seed>/` dirs hold
  `model.safetensors`, `config.json`, `tokenizer.json`.
- Final held-out perplexities in [15, 40]; no NaN losses in any train log.

## On failure

- A single OOM: rerun that one run at batch size 128 (the others stay at 256).
- 5+ runs fail: stop and report — likely a systematic bug, not bad luck.
- Never tune the frozen hyperparameters to "fix" perplexity; debug the code.
