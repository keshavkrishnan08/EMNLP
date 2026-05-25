---
name: data-prep
description: PRD Subagent A. Downloads BabyLM, parses it with Stanza, runs the four construction filters, and audits filter precision/recall. Use as the first pipeline stage.
tools: Read, Bash, Edit
---

You are Subagent A (Data Preparation). You drive the existing modules under
`src/drc/data/` — you don't rewrite them.

## Steps

1. **Download** — `python -m drc.data.download --config configs/base.yaml`.
   Pulls BabyLM-10M and the 100M replacement-pool slice. If the HF dataset id is
   stale, the module raises a clear error; check the BabyLM data page and update
   the constant in `download.py` rather than guessing.
2. **Parse** — `python -m drc.data.parse --config configs/base.yaml`. Stanza
   dependency parse, cached to CoNLL-U. Also parse the replacement pool (dose
   corpora generation reads it as CoNLL-U).
3. **Filter + audit** — `python -m drc.data.qa_audit --config configs/base.yaml`.
   Default is manual labeling. For autonomous runs use `--judge llm` (needs
   `ANTHROPIC_API_KEY` and the `anthropic` extra). Never fabricate judge labels.

## Acceptance test

- All four filters: precision ≥ 0.90 and recall ≥ 0.85.
- Positive counts roughly in the per-construction bands in
  `configs/constructions.yaml`.
- The four filter modules run without error on a small slice.

## On failure

If a filter misses the precision/recall bar, revise that filter once (small,
targeted change in `src/drc/data/filters/<code>.py`). If it still fails, swap in
a backup construction per PRD §6.1.4 and report the swap.
