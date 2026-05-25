"""Sanity tests for the generated Kaggle notebooks.

These don't run the notebooks (that needs Kaggle's GPUs); they just guard
against shipping a broken artifact. Each notebook must be valid nbformat v4
JSON with at least one cell, and every code cell's Python must actually parse —
catching the classic "edited the JSON by hand and broke a quote" mistake.

Jupyter magics (``%pip ...``) and shell lines (``!nvidia-smi``) aren't valid
Python on their own, so we strip those lines before parsing.
"""

from __future__ import annotations

import ast
import glob
import json
from pathlib import Path

import pytest

_NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
_NOTEBOOK_PATHS = sorted(glob.glob(str(_NOTEBOOKS_DIR / "*.ipynb")))


def _strip_magics(source: str) -> str:
    """Drop lines that are Jupyter magics or shell escapes, keeping line numbers.

    A line whose first non-space character is ``%`` or ``!`` is a magic/shell
    line, not Python. We blank it out (rather than delete it) so ``ast.parse``
    error line numbers still line up with the original cell.
    """
    out = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            out.append("")  # keep the line slot, drop the content
        else:
            out.append(line)
    return "\n".join(out)


def test_notebooks_exist():
    assert _NOTEBOOK_PATHS, f"no notebooks found under {_NOTEBOOKS_DIR}"


@pytest.mark.parametrize("path", _NOTEBOOK_PATHS, ids=lambda p: Path(p).name)
def test_notebook_is_valid(path):
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)  # must be valid JSON

    assert nb.get("nbformat") == 4, f"{path}: nbformat must be 4"
    cells = nb.get("cells")
    assert isinstance(cells, list) and cells, f"{path}: cells must be a non-empty list"

    for i, cell in enumerate(cells):
        assert cell.get("cell_type") in ("code", "markdown", "raw"), (
            f"{path}: cell {i} has an unknown cell_type {cell.get('cell_type')!r}"
        )
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        cleaned = _strip_magics(source)
        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            raise AssertionError(
                f"{path}: code cell {i} is not valid Python: {exc}"
            ) from exc
