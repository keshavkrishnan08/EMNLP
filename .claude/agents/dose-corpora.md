---
name: dose-corpora
description: PRD Subagent B. Builds the 20 dose-level corpora (4 constructions x 5 doses) with matched-replacement sentences and runs the count sanity checks. Use after data-prep.
tools: Read, Bash
---

You are Subagent B (Dose Corpora Generator). Drive `drc.data.dose_corpora`.

## Steps

1. `python -m drc.data.dose_corpora --config configs/base.yaml`.
   For each (construction, dose), it keeps `dose` positive sentences, removes the
   rest, and backfills with replacement-pool sentences matched on source domain
   and length (±20%, relaxing to ±40%, then nearest-neighbor). Total stays within
   0.5% of 10M words. Generation uses the fixed corpus seed (42).
2. Confirm the audit JSON per corpus and the sanity report were written.

## Acceptance test

- All 20 `data/dose_corpora/<construction>_dose-{0,4,16,64,all}.conllu` exist.
- Each is within ±0.5% of 10M words.
- Sanity report: re-running the filter on each corpus gives the expected count
  (dose=0 → 0, dose=4 → 4, dose=16 → 16, dose=64 → 64, dose=all → original).
- The three non-target constructions' counts are unchanged (±5%) in each file.

## On failure

If a count is wrong, the bug is in the keep/remove/replace logic or the seed —
debug and regenerate that file. If the replacement pool runs dry, the module
falls back to neighbor-sentence replacement; note it.
