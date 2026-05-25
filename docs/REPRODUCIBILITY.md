# Reproducibility

This project is built to be re-run end to end. Here's the EMNLP-style checklist, the command sequence to reproduce from scratch, and a clear line between what's deterministic and what's stochastic.

> **Status.** The pipeline is complete and runnable, but the experiments need GPUs (the sweep targets Kaggle dual-T4) and haven't been run in this repo. So `results/` ships empty and the paper's number slots are explicit placeholders. Nothing here is a fabricated result.

## Checklist

- [x] **Hyperparameters reported.** Every training hyperparameter is in `configs/base.yaml`, frozen across all 60 runs, and tabulated in `docs/METHODS.md` §3.
- [x] **Seeds fixed and reported.** Training seeds are **42, 43, 44** (`drc.SEEDS`). The corpus-generation seed is **42** (`CORPUS_SEED`), held separate from training randomness.
- [x] **Hardware reported.** Kaggle **2× T4**. Sixty runs, ~13 min each, ~11 h total wall time.
- [x] **Datasets public.** BabyLM-10M (plus a 100M replacement pool) from HuggingFace; see `data/README.md` for retrieval. Evaluation minimal pairs are hand-authored and shipped in `evaluation/eval_items/*.jsonl`.
- [x] **Code MIT-licensed.** See `LICENSE`. (BabyLM and external eval sets carry their own licenses.)
- [x] **Filter precision/recall reported.** Audited via `drc.data.qa_audit` against the bar precision ≥ 0.90, recall ≥ 0.85, per construction.
- [x] **Bootstrap iterations reported.** 1000 parametric-bootstrap resamples for the Hill CIs (`N_BOOTSTRAP = 1000`).
- [x] **Hypotheses pre-registered.** H1–H5 and the deflationary / composite outcomes are fixed in `docs/PRD.md` §4–5 *before* fitting.
- [x] **Decision rules pre-registered.** Exact thresholds live as named constants in `drc.analysis.decision`; documented in `docs/METHODS.md` §6.
- [x] **All figures script-generated.** Every figure is produced by `drc.analysis.figures` from the result CSVs — no manual numbers.
- [x] **Single source of truth for constants.** Construction codes, dose levels, and seeds are defined once in `src/drc/__init__.py` and imported everywhere.
- [x] **Tests for dependency-light logic.** `pytest` covers the filters, n-gram math, Hill curve math, package constants, and eval-item schema — runs without the ML stack.

## How to reproduce from scratch

Each stage is a module with its own CLI. End to end:

```bash
# 0. Install (ML stack only needed for train/eval; rest runs bare)
pip install -e ".[dev]"

# 1. Data: download, parse, audit filters, build the 20 dose corpora
python -m drc.data.download       --config configs/base.yaml
python -m drc.data.parse          --config configs/base.yaml
python -m drc.data.qa_audit       --config configs/base.yaml --judge manual
python -m drc.data.dose_corpora   --config configs/base.yaml

# 2. Tokenizer — trained once, reused by every run
python -m drc.tokenizer.train_tokenizer --config configs/base.yaml

# 3. Train — the pilot run gates the rest
python -m drc.train.sweep --config configs/base.yaml              # dual-T4
# python -m drc.train.sweep --config configs/base.yaml --single-gpu

# 4. Evaluate every model on all four test sets, plus the n-gram control
python -m drc.eval.run_eval       --config configs/base.yaml
python -m drc.eval.ngram_baseline --config configs/base.yaml

# 5. Fit curves, compare shapes, cluster, decide, draw figures
python -m drc.analysis.hill             --config configs/base.yaml
python -m drc.analysis.model_comparison --config configs/base.yaml
python -m drc.analysis.clustering       --config configs/base.yaml
python -m drc.analysis.decision         --config configs/base.yaml
python -m drc.analysis.figures          --config configs/base.yaml
```

`make pipeline` runs the lot in order. Kaggle launchers (dual-T4 and single-T4) are in `scripts/`.

## Deterministic vs. stochastic

This split is deliberate, and it's what makes the dose-response comparison clean.

**Deterministic — fixed once.**
- **Dose corpora.** Generated under corpus seed 42 and frozen. Same sentences kept, removed, and swapped in every time. The per-corpus audit JSON records exactly what happened, and the sanity check proves the realised counts (dose 0 → 0, dose 4 → 4, and so on).
- **Tokenizer.** One 16k BPE tokenizer, trained once, reused everywhere — vocabulary never drifts across the sweep.
- **Training config.** Every hyperparameter frozen across all 60 runs. No per-construction or per-dose tuning.

**Stochastic — varies across the three seeds.**
- **Model initialisation** and **data ordering** during training vary with seed 42 / 43 / 44. That's the *only* source of run-to-run variation, which is exactly why H5 (seed variance beats dose) is a clean, testable hypothesis: the data is identical across seeds, so any wobble is the model's, not the corpus's.

In short: the data is byte-for-byte reproducible; the model isn't, and we measure that gap on purpose.
