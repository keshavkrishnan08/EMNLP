"""Generate the four Kaggle dual-T4 notebooks for the DRC pipeline.

This is the single source of truth for the notebooks. The ``.ipynb`` files in
this directory are *build artifacts* — never hand-edit them. Change the cell
text here, re-run ``python notebooks/_build.py``, and the JSON is regenerated.

Why a generator instead of editing JSON by hand? Four notebooks share almost
all of their boilerplate (setup, artifact restore, GPU detect, status dump).
Factoring that shared text into Python functions keeps the four files honest
with each other: fix a bug in the setup cell once and every notebook gets it.

Each notebook is built on top of the fault-tolerant runner in
``drc.pipeline`` — we do not reinvent orchestration here. The notebooks just
make the repo importable on Kaggle, restore any prior artifacts so already-done
stages resume as skipped, detect the GPU count, and call ``run_pipeline``.

We prefer ``nbformat`` when it's installed (it validates as it writes) and fall
back to assembling the v4 JSON dict by hand otherwise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent

# The working directory on Kaggle. Everything — the cloned/installed repo, the
# config's relative paths, and all stage outputs — lives under here.
WORK_DIR = "/kaggle/working/drc-emnlp-2026"


# --------------------------------------------------------------------------- #
# Cell helpers
# --------------------------------------------------------------------------- #
def md(text: str) -> dict:
    """A markdown cell. ``text`` is stored as a list of lines (nbformat style)."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _as_source_lines(text),
    }


def code(text: str) -> dict:
    """A code cell. Outputs start empty; execution_count is null until run."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _as_source_lines(text),
    }


def _as_source_lines(text: str) -> list[str]:
    """Split source into newline-terminated lines, the way nbformat stores it.

    Every line keeps its trailing ``\\n`` except the last, matching how Jupyter
    serialises cell source. This makes diffs of the generated JSON readable.
    """
    text = text.strip("\n")
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        return lines
    # splitlines(keepends=True) drops a trailing newline edge case; normalise.
    return [ln if ln.endswith("\n") else ln + "\n" for ln in lines[:-1]] + (
        [lines[-1]] if lines else []
    )


# --------------------------------------------------------------------------- #
# Shared cell text
# --------------------------------------------------------------------------- #
def intro_md(title: str, what: str, only_note: str) -> dict:
    """The title/intro markdown shared by every notebook."""
    return md(
        f"""
# {title}

{what}

**This notebook is resumable.** Every stage runs in its own subprocess via
`drc.pipeline.run_pipeline`, so a CUDA OOM or a hard kernel crash in one stage
takes down that subprocess and *nothing else* — the kernel survives, the
failure is recorded, and the rest of the pipeline proceeds where it can. If
Kaggle's 12-hour session limit cuts you off, just **re-run the notebook**:
finished stages detect their own outputs and skip, so you pick up where you
stopped instead of redoing hours of work.

{only_note}

## Before you run

- Set the accelerator to **GPU T4 x2** (Settings -> Accelerator -> *GPU T4 x2*).
  With two cards the training sweep runs two jobs in parallel; with one it falls
  back to a 48-run single-GPU plan.
- **T4 is a Turing GPU and does not support bf16.** The shipped
  `configs/base.yaml` sets `precision: bf16`. Before training on T4, change that
  line to `precision: fp16`. The GPU-detect cell below reminds you; it does not
  edit the config for you.

## Chaining notebooks on Kaggle

The pipeline is split across notebooks so a long run survives the session limit
and each phase can be re-run on its own. They chain through Kaggle's
**notebook-output datasets**:

1. Run `kaggle_00_data` to completion. Its `/kaggle/working` is saved as a
   notebook-output dataset when the run finishes (*Save Version*).
2. In the next notebook (`kaggle_01_train`), add that output dataset as an input
   (*Add Input -> Your Datasets*). It lands under `/kaggle/input/<dataset>/`.
3. The **restore-prior-artifacts** cell copies any `data/`, `models/`, and
   `results/` it finds under `/kaggle/input/*/` into the working repo, so the
   already-done stages resume as *skipped* and this notebook only does its part.
4. Repeat: `kaggle_01_train`'s output feeds `kaggle_02_eval_analysis`.

Nothing here fabricates results. A blocked or failed stage produces no numbers —
it just says so in the dashboard and lets the rest proceed.
"""
    )


def setup_code(extras: str, pip_packages: str, extras_comment: str) -> dict:
    """The robust, idempotent environment-setup cell.

    ``extras`` is the package extra to try (e.g. ``"train"``); ``pip_packages``
    is a space-separated fallback/extra list of plain pip packages this notebook
    needs on top of the base ``drc`` install.
    """
    return code(
        f'''
# --- Setup: make the `drc` package importable, idempotently. -------------------
# Safe to re-run. If `drc` already imports we do nothing heavy. Otherwise we find
# the repo (uploaded as a dataset, or already cloned) or clone it, then install.
import os, sys, glob, subprocess
from pathlib import Path

# Replace this with your repo's clone URL (used only if the repo isn't already
# present as a Kaggle dataset or a prior checkout under /kaggle/working).
REPO_URL = "https://github.com/your-org/drc-emnlp-2026.git"  # <-- EDIT ME
WORK = Path("{WORK_DIR}")

# {extras_comment}
EXTRA = "{extras}"               # package extra to try, or "" for none
PIP_PACKAGES = "{pip_packages}"  # plain pip fallbacks this notebook needs


def _have_drc() -> bool:
    try:
        import drc  # noqa: F401
        return True
    except Exception:
        return False


def _find_repo() -> Path | None:
    """Look for an existing checkout: a uploaded dataset or a prior /kaggle/working."""
    candidates = []
    # A repo uploaded as a Kaggle dataset shows up under /kaggle/input/<name>/...
    candidates += glob.glob("/kaggle/input/*/drc-emnlp-2026")
    candidates += glob.glob("/kaggle/input/*/src/drc")   # repo root contains src/drc
    candidates += [str(WORK)]
    for c in candidates:
        c = Path(c)
        # Normalise: we want the repo ROOT (the dir that contains src/drc).
        root = c
        if root.name == "drc" and root.parent.name == "src":
            root = root.parent.parent
        if (root / "src" / "drc").exists() or (root / "pyproject.toml").exists():
            return root
    return None


def _run(cmd: str) -> int:
    print("$", cmd)
    return subprocess.call(cmd, shell=True)


if _have_drc():
    print("drc already importable — skipping repo setup.")
    # Still try to locate WORK so os.chdir below lands somewhere sensible.
    found = _find_repo()
    if found is not None:
        WORK = found
else:
    repo = _find_repo()
    if repo is not None and repo != WORK:
        print(f"Found repo at {{repo}} (not cloning).")
        WORK = repo
    elif repo is None:
        print(f"No local repo found; cloning {{REPO_URL}} -> {{WORK}}")
        WORK.parent.mkdir(parents=True, exist_ok=True)
        _run(f'git clone --depth 1 "{{REPO_URL}}" "{{WORK}}"')
    else:
        print(f"Using existing checkout at {{WORK}}")

    # Try an editable install with the extra; degrade gracefully on any failure.
    installed = False
    if EXTRA:
        rc = subprocess.call(
            f'pip install -q -e "{{WORK}}"[{{EXTRA}}]', shell=True
        )
        installed = rc == 0
        if not installed:
            print(f"[warn] editable install with [{{EXTRA}}] failed; trying plain -e")
    if not installed:
        rc = subprocess.call(f'pip install -q -e "{{WORK}}"', shell=True)
        installed = rc == 0
    if not installed:
        # Last resort: don't install, just put src/ on the path so imports work.
        src = str(WORK / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        print(f"[warn] pip install failed; added {{src}} to sys.path as fallback.")

    # Notebook-specific plain packages (e.g. stanza, scipy) on top of the base.
    if PIP_PACKAGES.strip():
        _run(f"pip install -q {{PIP_PACKAGES}}")

# Work from the repo root so the config's RELATIVE paths resolve under it.
os.chdir(WORK)
print("cwd:", os.getcwd())
print("drc importable:", _have_drc())
'''
    )


def restore_code() -> dict:
    """Copy prior-notebook artifacts into the working repo so stages resume."""
    return code(
        '''
# --- Restore prior artifacts so already-done stages resume as "skipped". -------
# When this notebook is chained after another, the previous notebook's
# /kaggle/working is attached as an input dataset under /kaggle/input/<name>/.
# We copy its data/ models/ results/ into our working repo. Safe to re-run;
# dirs_exist_ok lets it merge over an existing tree. Guarded so a missing input
# (e.g. when you run this notebook standalone) is a no-op, not an error.
import glob, shutil
from pathlib import Path

WORK = Path(os.getcwd())  # set by the setup cell
RESTORE_DIRS = ("data", "models", "results")

# Candidate source roots: a chained notebook-output dataset will contain the
# repo's working tree, either at the repo root or one level down.
sources = []
sources += glob.glob("/kaggle/input/*/drc-emnlp-2026")
sources += glob.glob("/kaggle/input/*")

restored = []
for src_root in sources:
    src_root = Path(src_root)
    if src_root.resolve() == WORK.resolve():
        continue
    for sub in RESTORE_DIRS:
        src = src_root / sub
        if src.is_dir():
            try:
                shutil.copytree(src, WORK / sub, dirs_exist_ok=True)
                restored.append(str(src))
            except Exception as exc:  # never let a restore failure stop the run
                print(f"[warn] could not restore {src}: {exc}")

if restored:
    print("Restored prior artifacts from:")
    for r in restored:
        print("  ", r)
else:
    print("No prior artifacts found to restore (fine if this is the first stage).")
'''
    )


def gpu_detect_code() -> dict:
    """Detect GPU count and set SINGLE_GPU; warn about the T4 bf16 caveat."""
    return code(
        '''
# --- Detect GPUs and pick the sweep mode. --------------------------------------
import subprocess

n_gpus = 0
try:
    out = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
    )
    print(out.stdout.strip() or "(nvidia-smi returned no GPUs)")
    n_gpus = sum(1 for ln in out.stdout.splitlines() if ln.strip().startswith("GPU "))
except FileNotFoundError:
    print("nvidia-smi not found — assuming no GPU (CPU-only).")

SINGLE_GPU = n_gpus < 2
print(f"\\nDetected {n_gpus} GPU(s). SINGLE_GPU = {SINGLE_GPU}")

if n_gpus >= 2:
    print("Dual-GPU: the sweep runs two training jobs in parallel, one per card.")
elif n_gpus == 1:
    print("Single-GPU: the sweep uses the 48-run fallback (drops dose=4).")
else:
    print("No GPU: training/eval stages will fail; set Accelerator to GPU T4 x2.")

print(
    "\\n[CAVEAT] T4 is a Turing card and does NOT support bf16. The shipped"
    "\\n         configs/base.yaml uses precision: bf16. Before training on T4,"
    "\\n         edit that line to  precision: fp16  (this cell does not edit it)."
)
'''
    )


def run_code(only_expr: str, only_human: str) -> dict:
    """The cell that actually calls the pipeline runner."""
    return code(
        f'''
# --- Run the pipeline. ---------------------------------------------------------
# default_phases() returns the 13 wired stages; run_pipeline() isolates failures,
# skips finished stages, blocks stages with unmet deps, and prints a dashboard.
# It never raises on a stage failure, so this cell completes even if a stage dies.
from pathlib import Path
from drc.pipeline import default_phases, run_pipeline

cfg = Path("configs/base.yaml")
status = Path("results/pipeline_status.json")

# {only_human}
only = {only_expr}

stages = default_phases(cfg, single_gpu=SINGLE_GPU)
results = run_pipeline(stages, status_path=status, only=only)
# The dashboard is already printed above by run_pipeline.
'''
    )


def status_code() -> dict:
    """Pretty-print the status JSON and list result artifacts; tolerant of gaps."""
    return code(
        '''
# --- Inspect what we produced. -------------------------------------------------
# Tolerant of missing files: a fresh or partial run just shows fewer artifacts.
import json
from pathlib import Path

status_path = Path("results/pipeline_status.json")
if status_path.exists():
    data = json.loads(status_path.read_text())
    print("Pipeline status:")
    for name, r in data.items():
        secs = f"{r.get('seconds', 0):.1f}s" if r.get("seconds") else ""
        detail = f"  {r['detail']}" if r.get("detail") else ""
        print(f"  {r['status']:<8} {name:<18} {secs}{detail}")
else:
    print("No results/pipeline_status.json yet — has the run cell completed?")

for d in ("results", "results/figures"):
    p = Path(d)
    if p.is_dir():
        items = sorted(x.name for x in p.iterdir())
        print(f"\\n{d}/ ({len(items)} items):")
        for it in items:
            print("  ", it)
    else:
        print(f"\\n{d}/ does not exist yet.")

decision = Path("results/decision.txt")
if decision.exists():
    print("\\n=== results/decision.txt ===")
    print(decision.read_text())
'''
    )


# --------------------------------------------------------------------------- #
# Notebook assembly
# --------------------------------------------------------------------------- #
def build_notebook(cells: list[dict]) -> dict:
    """Wrap a list of cells into a complete nbformat v4 notebook dict."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dockerImageVersionId": 0},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _assign_cell_ids(cells: list[dict]) -> None:
    """Give each cell a stable `id` (required by nbformat 4.5+).

    Deterministic — derived from the cell's index and source — so re-running the
    generator produces byte-identical ids and clean diffs.
    """
    for i, cell in enumerate(cells):
        digest = hashlib.sha1(
            (str(i) + "".join(cell["source"])).encode("utf-8")
        ).hexdigest()
        cell["id"] = digest[:12]


def write_notebook(path: Path, cells: list[dict]) -> None:
    """Write the notebook, validating via nbformat when it's available."""
    _assign_cell_ids(cells)
    nb = build_notebook(cells)
    try:
        import nbformat

        nb_obj = nbformat.from_dict(nb)
        nbformat.validate(nb_obj)
        with open(path, "w", encoding="utf-8") as fh:
            nbformat.write(nb_obj, fh)
    except ImportError:
        # Hand-rolled v4 JSON — still valid, just not validated.
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1)
            fh.write("\n")
    print("wrote", path)


# Comments describing the install profile for each notebook's setup cell.
_TRAIN_COMMENT = "Training needs the ML stack: torch, transformers, accelerate, etc."
_DATA_COMMENT = "Data stages need Stanza (parsing) and datasets (download)."
_EVAL_COMMENT = "Eval needs torch; analysis needs scipy/sklearn/matplotlib for fits + figures."


def make_run_all() -> list[dict]:
    return [
        intro_md(
            "DRC — Run the whole pipeline (single session)",
            "This is the **master** notebook. It runs all 13 stages end to end: "
            "download, parse, audit, dose, tokenizer, train, eval, ngram, hill, "
            "model_comparison, clustering, decision, figures. Use it when you "
            "want everything in one go and expect to fit inside one session (or "
            "to re-run after a timeout and let resume carry you the rest of the "
            "way). For long training, the split notebooks (00/01/02) chain more "
            "comfortably across sessions.",
            "**Scope:** every stage (`only=None`).",
        ),
        # The master needs everything: ML stack + data deps + analysis deps.
        setup_code(
            extras="train",
            pip_packages="stanza datasets scipy scikit-learn matplotlib",
            extras_comment="Master notebook: install the full stack (train + data + analysis).",
        ),
        restore_code(),
        gpu_detect_code(),
        run_code("None  # None == run every stage", "Run all stages: only=None."),
        status_code(),
    ]


def make_data() -> list[dict]:
    return [
        intro_md(
            "DRC — Data preparation (download -> dose corpora)",
            "Builds the corpora the rest of the pipeline depends on. Stages: "
            "**download, parse, audit, dose, tokenizer**. This is the first link "
            "in the chain — run it to completion, *Save Version*, and attach its "
            "output to `kaggle_01_train`. No GPU is strictly required here, but "
            "Stanza parsing is much faster with one.",
            "**Scope:** `only=[\"download\", \"parse\", \"audit\", \"dose\", \"tokenizer\"]`.",
        ),
        setup_code(
            extras="train",
            pip_packages="stanza datasets",
            extras_comment=_DATA_COMMENT,
        ),
        # 00 is the first notebook, so there's usually nothing to restore — but
        # we keep the cell so re-running after a partial run still merges cleanly.
        restore_code(),
        gpu_detect_code(),
        run_code(
            '["download", "parse", "audit", "dose", "tokenizer"]',
            "Data stages only.",
        ),
        status_code(),
    ]


def make_train() -> list[dict]:
    return [
        intro_md(
            "DRC — Training sweep (dual-T4, resumable)",
            "Runs the long **train** stage: the pilot-gated sweep over every "
            "construction x dose x seed (60 runs on dual-T4, 48 in single-GPU "
            "fallback). Each model trains in its own subprocess, and finished "
            "runs leave a `metrics.json` that marks them done — so a timeout or "
            "crash costs you at most the in-flight runs. Re-run to continue.",
            "**Scope:** `only=[\"train\"]`. Attach `kaggle_00_data`'s output first "
            "so the data stages restore and the sweep finds its corpora and "
            "tokenizer.",
        ),
        setup_code(
            extras="train",
            pip_packages="",
            extras_comment=_TRAIN_COMMENT,
        ),
        restore_code(),
        gpu_detect_code(),
        run_code('["train"]', "Training sweep only."),
        status_code(),
    ]


def make_eval_analysis() -> list[dict]:
    return [
        intro_md(
            "DRC — Evaluation & analysis (curves, clusters, figures)",
            "Turns trained models into results. Stages: **eval, ngram, hill, "
            "model_comparison, clustering, decision, figures**. Note `ngram` "
            "only needs the corpora, so it runs even if training never finished. "
            "`eval` and the analysis stages need the model manifest, so attach "
            "`kaggle_01_train`'s output (which also carries the data) before "
            "running.",
            "**Scope:** `only=[\"eval\", \"ngram\", \"hill\", \"model_comparison\", "
            "\"clustering\", \"decision\", \"figures\"]`.",
        ),
        setup_code(
            extras="train",  # eval imports torch
            pip_packages="scipy scikit-learn matplotlib",
            extras_comment=_EVAL_COMMENT,
        ),
        restore_code(),
        gpu_detect_code(),
        run_code(
            '["eval", "ngram", "hill", "model_comparison", '
            '"clustering", "decision", "figures"]',
            "Evaluation and analysis stages.",
        ),
        status_code(),
    ]


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "kaggle_run_all.ipynb": make_run_all(),
        "kaggle_00_data.ipynb": make_data(),
        "kaggle_01_train.ipynb": make_train(),
        "kaggle_02_eval_analysis.ipynb": make_eval_analysis(),
    }
    for name, cells in notebooks.items():
        write_notebook(NOTEBOOKS_DIR / name, cells)
    print(f"Generated {len(notebooks)} notebooks in {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
