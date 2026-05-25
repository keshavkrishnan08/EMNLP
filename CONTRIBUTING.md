# Contributing

Thanks for your interest. A few notes to keep the repo reproducible.

## Development setup

```bash
pip install -e ".[dev]"
pytest          # the suite runs without the GPU stack
ruff check src tests
```

## Adding a construction

The pipeline is built so a new construction is roughly an hour of work:

1. Add its code to `CONSTRUCTIONS` in `src/drc/__init__.py`.
2. Write a filter in `src/drc/data/filters/<code>.py` implementing the
   `ConstructionFilter` protocol from `filters/base.py`, and register it in
   `filters/__init__.py`.
3. Add a metadata block to `configs/constructions.yaml`.
4. Drop minimal-pair items at `evaluation/eval_items/<code>.jsonl`.
5. Add filter unit tests in `tests/test_filters.py`.

Everything downstream — dose corpora, training, evaluation, analysis — iterates
over `CONSTRUCTIONS`, so it picks up the new one automatically.

## Conventions

- Keep heavy dependencies (torch, transformers, stanza, datasets) imported
  *lazily* inside functions, so modules import in a bare environment.
- Use the module `logging` logger for progress, not `print`.
- Don't tune the training hyperparameters in `configs/base.yaml` — they're frozen
  on purpose (see the file's header comment).
- Never commit corpora, checkpoints, or fabricated results.

## Commit style

Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.
