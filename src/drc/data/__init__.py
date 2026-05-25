"""Corpus pipeline: download, parse, dose-level generation, and QA.

The four stages here turn the raw BabyLM release into the 20 dose-level
corpora the sweep trains on. They run in order:

1. ``download``  — pull BabyLM-10M and a 100M replacement pool to plain text.
2. ``parse``     — run Stanza once and cache CoNLL-U to disk.
3. ``dose_corpora`` — thin each construction down to 0/4/16/64/all instances.
4. ``qa_audit``  — sample filter hits/misses for a precision/recall check.

Each module ships its own argparse CLI, so you can run a single stage with
``python -m drc.data.<stage> --config configs/base.yaml``.
"""

from __future__ import annotations
