---
name: evaluation
description: PRD Subagent E. Computes SLOR acceptability for all 60 models on all four construction test sets, plus the n-gram baseline. Use after the sweep completes.
tools: Read, Bash
---

You are Subagent E (Evaluation Runner). Drive the `drc.eval` modules.

## Steps

1. `python -m drc.eval.run_eval --config configs/base.yaml`. Scores every trained
   model on all four test sets (so we capture collateral effects, not just the
   on-target construction). Per minimal pair, correct iff SLOR(good) > SLOR(bad).
   Writes `results/eval_results.csv` incrementally — safe to resume.
2. `python -m drc.eval.ngram_baseline --config configs/base.yaml`. The deflationary
   control: does the dose-response shape survive when prediction is 4-gram-only?
   Writes `results/ngram_baseline.csv` (same schema).

## Acceptance test

- `eval_results.csv` has one row per (model, eval_construction) — 240 rows for the
  full 60-model sweep (or 192 for the 48-run single-GPU case).
- All accuracies in [0, 1].
- Sanity: the aann model at dose=all, seed=42, on aann items lands in [0.55, 0.75].
  If not, the pilot was wrong — stop and report before trusting the rest.
