"""Validate the evaluation minimal-pair files against the expected schema.

Skips cleanly when no eval_items/*.jsonl files are present yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import drc

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = _REPO_ROOT / "evaluation" / "eval_items"

REQUIRED_KEYS = ("item_id", "construction", "good_sentence", "bad_sentence")


def _jsonl_files() -> list[Path]:
    if not _EVAL_DIR.is_dir():
        return []
    return sorted(_EVAL_DIR.glob("*.jsonl"))


_FILES = _jsonl_files()

if not _FILES:
    pytest.skip(
        "no evaluation/eval_items/*.jsonl files present", allow_module_level=True
    )


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_eval_file_has_valid_lines(path: Path):
    """Every line parses, carries the required keys, and names a real code."""
    seen_any = False
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            seen_any = True
            obj = json.loads(line)

            for key in REQUIRED_KEYS:
                assert key in obj, f"{path.name}:{lineno} missing key {key!r}"

            construction = obj["construction"]
            assert construction in drc.CONSTRUCTIONS, (
                f"{path.name}:{lineno} unknown construction {construction!r}"
            )
            assert obj["good_sentence"].strip()
            assert obj["bad_sentence"].strip()
            assert obj["good_sentence"] != obj["bad_sentence"]

    assert seen_any, f"{path.name} contained no items"
