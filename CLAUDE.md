# CLAUDE.md — DRC-EMNLP-2026

Project-specific guidance for agents working in this folder. This overrides the
root workspace `CLAUDE.md` where they conflict.

## What this project is

A from-scratch dose-response study of grammatical construction learning. Train
LTG-BERT on BabyLM-10M at five exposure levels per construction, fit Hill curves
to SLOR acceptability, decide among five pre-registered hypotheses. Design lives
in [`docs/PRD.md`](docs/PRD.md); methods in [`docs/METHODS.md`](docs/METHODS.md).

## Architecture in one breath

`src/drc/` is an installable package. Each pipeline stage is a module with a CLI
(`python -m drc.<stage>`). Stages communicate through the filesystem with fixed
naming (see README "Naming conventions"), so any stage can find the previous
one's output and the sweep can resume after a timeout.

```
data.download → data.parse → data.filters → data.qa_audit → data.dose_corpora
  → tokenizer.train_tokenizer → train.sweep → eval.run_eval / eval.ngram_baseline
  → analysis.{hill,model_comparison,clustering,decision,figures}
```

## Hard rules

- **Heavy deps are lazy.** torch, transformers, stanza, datasets, tokenizers,
  anthropic — import them *inside functions*, never at module top level. The
  package and the test suite must import in a bare environment.
- **Frozen hyperparameters.** `configs/base.yaml` training block is fixed. If a
  pilot underperforms, debug the implementation, not the config.
- **Never fabricate results.** No invented accuracies, Hill parameters, figures,
  or paper findings. `results/` stays empty until a real run fills it; the paper
  uses visible `\resultpending` placeholders.
- **Constants live once.** Construction codes, dose levels, and seeds are defined
  in `src/drc/__init__.py`. Import them; don't re-list them.
- **Tests and lint must pass.** `pytest` and `ruff check src tests` before any
  commit. CI runs both.

## Prose style (docstrings, comments, docs, paper)

Active voice. Vary sentence length. Occasional contractions. Explain *why*, not
just what. No corporate filler. This applies to user-facing text and code
comments alike — the repo should read like a careful human wrote it.

## 符 Token Glossary

Per the workspace convention, use these single-character shorthands in
*inter-agent messages, logs, and internal reasoning only*. Always expand to
English in committed code, docstrings, docs, and the paper.

| 符 | Expansion |
|----|-----------|
| 构 | construction |
| 剂 | dose / exposure level |
| 域 | dose-response curve |
| 拟 | Hill-equation fit |
| 评 | SLOR evaluation |
| 滤 | construction filter |
| 语 | BabyLM-10M corpus |
| 训 | training run (one model) |
| 扫 | the 60-run sweep |
| 判 | pre-registered decision rule |
| 验 | acceptance test |
| 种 | random seed (42/43/44) |
