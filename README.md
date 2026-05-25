# Dose-Response Curves for Construction Learning

*A scaling law for grammatical acquisition in small language models.*

Prior work on how language models learn rare grammatical constructions treats
exposure as a switch: the construction is either in the training corpus or
filtered out ([Misra & Mahowald 2024](https://aclanthology.org/2024.emnlp-main.53/);
[Patil et al. 2024](https://doi.org/10.1162/tacl_a_00720)). This project turns that
switch into a dial. We train LTG-BERT models from scratch on BabyLM-10M with a
target construction present at logarithmically spaced doses — 0, 4, 16, 64, and
every attested instance — then fit parametric **dose-response curves** to the
resulting acceptability judgments. The fitted parameters (threshold `E50`, Hill
slope `n`, floor `E0`, ceiling `Emax`) become a quantitative theory of how much
direct evidence a construction needs.

The full design, hypotheses, and rationale live in [`docs/PRD.md`](docs/PRD.md).

> **Status.** This repository contains the complete, runnable pipeline. The
> experiments require GPUs (the sweep is built for Kaggle dual-T4) and have not
> been run here, so `results/` ships empty and the paper's result slots are
> explicit placeholders. Nothing in this repo is a fabricated number.

## What's in the box

| Stage | Module | What it does |
|-------|--------|--------------|
| Download | `drc.data.download` | Pull BabyLM-10M + a replacement pool from HuggingFace |
| Parse | `drc.data.parse` | Stanza dependency parse → cached CoNLL-U |
| Filter | `drc.data.filters` | Four construction detectors (AANN, comparative correlative, tough-movement, resultative) |
| QA audit | `drc.data.qa_audit` | Precision/recall against a human or LLM judge |
| Dose corpora | `drc.data.dose_corpora` | Build the 20 dose-level corpora + sanity checks |
| Tokenizer | `drc.tokenizer.train_tokenizer` | One frozen 16k byte-level BPE tokenizer |
| Train | `drc.train.train` / `drc.train.sweep` | One run, or the full 60-run sweep (pilot-gated) |
| Evaluate | `drc.eval.run_eval` / `drc.eval.ngram_baseline` | SLOR acceptability + n-gram control |
| Analyze | `drc.analysis.*` | Hill fits, model comparison, clustering, decision rule, figures |

The four constructions span productivity, frequency, and theoretical priority,
so heterogeneity in their curves can't collapse to a single confound. Five
pre-registered hypotheses (H1–H5) each map to a distinct paper framing, chosen
*after* the decision rule fires — see [`docs/PRD.md`](docs/PRD.md) §3 and
`drc.analysis.decision`.

## Install

```bash
git clone <your-fork-url> drc-emnlp-2026
cd drc-emnlp-2026
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # or: pip install -r requirements.txt
```

Heavy dependencies (PyTorch, Transformers, Stanza, `datasets`) are imported
lazily, so the package imports and the test suite run in a bare environment.
You only need the GPU stack for the training and evaluation stages.

## Run the pipeline

Each stage is a module with its own CLI. End to end:

```bash
# 1. Data: download, parse, filter, audit, build dose corpora
python -m drc.data.download       --config configs/base.yaml
python -m drc.data.parse          --config configs/base.yaml
python -m drc.data.qa_audit       --config configs/base.yaml --judge manual
python -m drc.data.dose_corpora   --config configs/base.yaml

# 2. Tokenizer (trained once, reused by every run)
python -m drc.tokenizer.train_tokenizer --config configs/base.yaml

# 3. Train: pilot runs first and gates the rest
python -m drc.train.sweep --config configs/base.yaml            # dual-GPU
python -m drc.train.sweep --config configs/base.yaml --single-gpu

# 4. Evaluate every model on all four test sets, plus the n-gram baseline
python -m drc.eval.run_eval       --config configs/base.yaml
python -m drc.eval.ngram_baseline --config configs/base.yaml

# 5. Fit curves, compare shapes, cluster, decide, draw figures
python -m drc.analysis.hill             --config configs/base.yaml
python -m drc.analysis.model_comparison --config configs/base.yaml
python -m drc.analysis.clustering       --config configs/base.yaml
python -m drc.analysis.decision         --config configs/base.yaml
python -m drc.analysis.figures          --config configs/base.yaml
```

`make pipeline` runs the lot in order. On Kaggle, see
[`scripts/`](scripts/) for the dual-T4 and single-T4 launchers.

## Naming conventions

Outputs follow predictable patterns so every stage can find the last one's work:

- Dose corpora: `data/dose_corpora/{construction}_dose-{dose}.conllu`
- Model runs: `models/ltgbert_{construction}_D{dose}_seed{seed}/`
- Results: flat CSVs under `results/` (`eval_results.csv`, `hill_fits.csv`, …)

Construction codes, dose levels, and seeds are defined once in
[`src/drc/__init__.py`](src/drc/__init__.py) and imported everywhere.

## Tests

```bash
pytest
```

The suite covers the dependency-light logic — construction filters (against
fake parses), the n-gram model, the Hill curve math, the package constants, and
the evaluation-item schema. It runs without the ML stack.

## Reproducibility

Hyperparameters are frozen in [`configs/base.yaml`](configs/base.yaml); seeds are
42/43/44; the dose corpora are generated under a fixed corpus seed. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full checklist and
[`data/README.md`](data/README.md) for how to obtain BabyLM.

## Citing

See [`CITATION.cff`](CITATION.cff). The manuscript source is in
[`paper/`](paper/).

## License

Code is released under the [MIT License](LICENSE). BabyLM and the external
evaluation sets carry their own licenses — see [`data/README.md`](data/README.md).
