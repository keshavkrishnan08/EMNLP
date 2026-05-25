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
WORK_DIR = "/kaggle/working/EMNLP"

# Default clone URL baked into the notebooks. HTTPS (not SSH) because Kaggle has
# no SSH deploy key — HTTPS clones a PUBLIC repo with no auth. If the repo is
# private, either make it public, upload it as a Kaggle dataset (the setup cell
# finds it under /kaggle/input automatically), or put a token in the URL.
REPO_URL = "https://github.com/keshavkrishnan08/EMNLP.git"


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
- **Precision is automatic.** The trainer picks bf16 on GPUs that support it,
  fp16 (with a GradScaler) on a T4, and fp32 on CPU — so you don't need to edit
  `configs/base.yaml`, even though it says `bf16`.

## The notebooks (split for a single-GPU, two-session run)

A single-T4 run is ~13-14 h, over Kaggle's 12 h cap, so the work is split so each
half fits one session:

1. **`kaggle_00_smoke`** — the whole pipeline on a tiny config (~10 min). Run it
   first to confirm every stage works before the long run.
2. **`kaggle_01a_data_train`** — session 1 (the long pole): download, parse,
   audit, dose corpora, tokenizer, and the 63-model training sweep. Ends with a
   health check on the trained models. If the 12 h cap cuts it off, just re-run —
   finished models are skipped, so it resumes.
3. **`kaggle_01b_eval_analysis`** — session 2 (~2-3 h): SLOR + n-gram evaluation,
   then all the analysis (Hill fits, E0 index, transfer, predictability,
   generalization, decision) and figures.

They chain through Kaggle's **notebook-output datasets**: run `01a` to completion,
*Save Version*, then in `01b` add that output as an input (*Add Input -> Your
Datasets*). The **restore-prior-artifacts** cell copies its `data/`, `models/`,
and `results/` into the working repo so `01b` finds the trained models.
(`kaggle_run_all` runs everything in one go, for reference / dual-GPU.)

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

# Clone URL, used only if the repo isn't already present (as a Kaggle dataset or
# a prior /kaggle/working checkout). HTTPS clones a PUBLIC repo with no auth; for
# a private repo, upload it as a Kaggle dataset instead (the cell finds it under
# /kaggle/input), or put a token in the URL: https://<token>@github.com/owner/repo.git
REPO_URL = "{REPO_URL}"
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
    # Match it however it's nested: a `src/drc` at one or two levels down, or a
    # named folder. We don't assume any particular dataset/repo name.
    candidates += glob.glob("/kaggle/input/*/src/drc")
    candidates += glob.glob("/kaggle/input/*/*/src/drc")
    candidates += glob.glob("/kaggle/input/*/EMNLP")
    candidates += glob.glob("/kaggle/input/*")
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


# Locate the repo: an existing checkout, an uploaded dataset, or a fresh clone.
# We do this even if `drc` already imports, because a stale checkout from an
# earlier run would otherwise pin you to old code (and silently re-run old bugs).
repo = _find_repo()
if repo is None:
    print(f"No local repo found; cloning {{REPO_URL}} -> {{WORK}}")
    WORK.parent.mkdir(parents=True, exist_ok=True)
    _run(f'git clone --depth 1 "{{REPO_URL}}" "{{WORK}}"')
else:
    WORK = repo
    print(f"Using repo at {{WORK}}")
    # If it's a writable git checkout, pull the latest so code fixes actually land.
    # This is the difference between re-running old buggy code and the current fix.
    # (Tracked files only — your gitignored data/ models/ results/ are untouched.)
    if (WORK / ".git").exists():
        rc = subprocess.call(
            f'git -C "{{WORK}}" fetch --depth 1 origin main '
            f'&& git -C "{{WORK}}" reset --hard origin/main', shell=True,
        )
        print("Updated checkout to latest origin/main."
              if rc == 0 else "[warn] could not update checkout; using it as-is.")

# Install — ALWAYS, whether we just cloned or found an existing checkout. (This
# block is intentionally at top level, not under the else above: a fresh clone
# must still install its deps.) Editable install registers `drc` and pulls deps;
# note the extras brackets go INSIDE the quotes — pip install -e "PATH[extra]" —
# or the shell splits off "[extra]" and pip errors.
installed = False
if EXTRA:
    installed = subprocess.call(f'pip install -q -e "{{WORK}}[{{EXTRA}}]"', shell=True) == 0
    if not installed:
        print("[warn] editable install with the extra failed; trying plain -e")
if not installed:
    installed = subprocess.call(f'pip install -q -e "{{WORK}}"', shell=True) == 0
if not installed:
    print("[warn] editable install failed; relying on PYTHONPATH below.")

# Notebook-specific plain packages Kaggle's image lacks (e.g. stanza). Installed
# explicitly so they land even if the editable-extra resolve was skipped.
if PIP_PACKAGES.strip():
    _run(f"pip install -q {{PIP_PACKAGES}}")

# Make `drc` importable BOTH here and in the subprocesses the pipeline spawns.
# The pipeline runs each stage as `python -m drc...`, a fresh process that
# inherits PYTHONPATH (not this cell's sys.path), so we set both. This is what
# makes the run work even if the editable install above didn't register.
SRC = str(WORK / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")

# Work from the repo root so the config's RELATIVE paths resolve under it.
if WORK.exists():
    os.chdir(WORK)
    print("cwd:", os.getcwd())
else:
    print("=" * 70)
    print(f"[error] {{WORK}} does not exist — the clone FAILED.")
    print("Almost certainly: Kaggle INTERNET IS OFF for this notebook.")
    print("Fix: Settings (right panel) -> Internet -> ON (needs a phone-verified")
    print("account), then re-run. The pipeline also downloads BabyLM from")
    print("HuggingFace, so internet is required regardless of the clone.")
    print("Alternatively, upload the repo as a Kaggle dataset (Add Input).")
    print("=" * 70)
print("drc importable:", _have_drc())
if not _have_drc():
    print("[error] `drc` not importable — see the message above (likely no internet).")
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
sources += glob.glob("/kaggle/input/*/EMNLP")
sources += glob.glob("/kaggle/input/*/*")
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
# Set this True to run SEQUENTIALLY on a single card even when two are present
# (e.g. on a T4 x2 box, to use the faster T4 fp16 path without the parallel
# orchestration). The sweep then trains one model at a time on GPU 0.
FORCE_SINGLE_GPU = False

import os
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

SINGLE_GPU = FORCE_SINGLE_GPU or (n_gpus < 2)
if FORCE_SINGLE_GPU and n_gpus >= 2:
    # Use one card only: pin the env so every training subprocess sees GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("FORCE_SINGLE_GPU set: using GPU 0 only, running sequentially.")
print(f"\\nDetected {n_gpus} GPU(s). SINGLE_GPU = {SINGLE_GPU}")

if SINGLE_GPU:
    print("Single-GPU: the sweep trains one model at a time on GPU 0 (no parallelism).")
else:
    print("Dual-GPU: the sweep runs two training jobs in parallel, one per card.")
if n_gpus == 0:
    print("No GPU: training/eval stages will fail; set Accelerator to GPU T4 x2.")

print(
    "\\n[note] Precision is auto-selected by the trainer: bf16 on GPUs that"
    "\\n       support it (Ampere+), fp16 on a T4 (with a GradScaler), fp32 on"
    "\\n       CPU. You do NOT need to edit configs/base.yaml for the T4."
)
'''
    )


def deps_code(packages: list[str]) -> dict:
    """An explicit, always-runs cell that installs the deps Kaggle's image lacks.

    Deliberately its own cell rather than buried in the setup logic: that way the
    install can't be skipped by a branch, you can see exactly what's installed,
    and you can add a package by editing one obvious list. Idempotent — pip skips
    anything already present.
    """
    pkgs = " ".join(packages)
    return code(
        f'''
# --- Install runtime dependencies Kaggle's image doesn't ship. -----------------
# `stanza` (used by the parse stage) is not on the Kaggle GPU image; torch,
# transformers, datasets, scipy, sklearn, matplotlib already are. This runs every
# time and is idempotent, so the parse stage can always `import stanza`.
import subprocess, sys

PACKAGES = "{pkgs}".split()
if PACKAGES:
    print("Installing:", PACKAGES)
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    print("pip exit code:", rc)
    for pkg in PACKAGES:
        mod = pkg.split("==")[0].split(">=")[0].replace("-", "_")
        try:
            __import__(mod)
            print(f"  import {{mod}}: OK")
        except Exception as exc:
            print(f"  import {{mod}}: FAILED — {{exc}}")
else:
    print("No extra packages to install.")
'''
    )


def run_code(only_expr: str, only_human: str, config_path: str = "configs/base.yaml") -> dict:
    """The cell that actually calls the pipeline runner.

    ``config_path`` selects the config (the smoke notebook passes a tiny one).
    Output locations are read from that config, so they're never hard-coded — the
    later cells reuse ``CONFIG_PATH`` to find the right results directory.
    """
    return code(
        f'''
# --- Run the pipeline. ---------------------------------------------------------
# default_phases() returns the wired stages; run_pipeline() isolates failures,
# skips finished stages, blocks stages with unmet deps, and prints a dashboard.
# It never raises on a stage failure. We also wrap the whole cell so that even a
# setup/import problem prints instead of halting — the status cells below still run.
import traceback
from pathlib import Path

CONFIG_PATH = "{config_path}"          # reused by the cells below
# {only_human}
only = {only_expr}

try:
    from drc.pipeline import default_phases, run_pipeline
    from drc.data.download import load_config, resolve_path

    cfg = Path(CONFIG_PATH)
    results_dir = resolve_path(cfg, load_config(cfg)["paths"]["results"])
    status = results_dir / "pipeline_status.json"
    stages = default_phases(cfg, single_gpu=SINGLE_GPU)
    results = run_pipeline(stages, status_path=status, only=only)
    # The dashboard is already printed above by run_pipeline.
except Exception:
    print("[error] the run cell hit an exception; the cells below will still run.")
    traceback.print_exc()
'''
    )


def status_code() -> dict:
    """Pretty-print the status JSON and list result artifacts; tolerant of gaps."""
    return code(
        '''
# --- Inspect what we produced. -------------------------------------------------
# Tolerant of missing files and of being run on its own: a fresh or partial run
# just shows fewer artifacts. Wrapped so it never halts a Run All.
import json
import traceback
from pathlib import Path

CONFIG_PATH = globals().get("CONFIG_PATH", "configs/base.yaml")
try:
    from drc.data.download import load_config, resolve_path

    results_dir = resolve_path(Path(CONFIG_PATH), load_config(CONFIG_PATH)["paths"]["results"])

    status_path = results_dir / "pipeline_status.json"
    if status_path.exists():
        data = json.loads(status_path.read_text())
        print("Pipeline status:")
        for name, r in data.items():
            secs = f"{r.get('seconds', 0):.1f}s" if r.get("seconds") else ""
            detail = f"  {r['detail']}" if r.get("detail") else ""
            print(f"  {r['status']:<8} {name:<18} {secs}{detail}")
    else:
        print(f"No {status_path} yet — has the run cell completed?")

    for d in (results_dir, results_dir / "figures"):
        if d.is_dir():
            items = sorted(x.name for x in d.iterdir())
            print(f"\\n{d}/ ({len(items)} items):")
            for it in items:
                print("  ", it)
        else:
            print(f"\\n{d}/ does not exist yet.")

    decision = results_dir / "decision.txt"
    if decision.exists():
        print("\\n=== decision.txt ===")
        print(decision.read_text())
except Exception:
    print("[error] status cell failed; continuing.")
    traceback.print_exc()
'''
    )


def results_summary_code() -> dict:
    """A 'do I need to re-run?' health check for the heavy results notebook.

    Reads the sweep manifest and eval CSV and prints, per construction, the
    zero-exposure floor E0 and the shared-ceiling E_max, plus any failed runs and
    out-of-band perplexities. The point is to decide whether the expensive run is
    trustworthy before moving on to analysis.
    """
    return code(
        '''
# --- Results health check: should I re-run anything? ---------------------------
# Tolerant of partial runs. Surfaces failed training runs, perplexity outliers,
# and the E0 / E_max each construction landed at, so you can judge the run.
import json
from pathlib import Path

CONFIG_PATH = globals().get("CONFIG_PATH", "configs/base.yaml")
from drc.data.download import load_config, resolve_path

_paths = load_config(CONFIG_PATH)["paths"]
results_dir = resolve_path(Path(CONFIG_PATH), _paths["results"])
models_dir = resolve_path(Path(CONFIG_PATH), _paths["models"])

problems = []

# 1) Training sweep: how many runs finished, and did any fail?
manifest = results_dir / "sweep_manifest.json"
if manifest.exists():
    m = json.loads(manifest.read_text())
    runs = m.get("runs", {})
    by_status = {}
    for r in runs.values():
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
    print("Training runs:", dict(by_status), f"(of {m.get('total_runs', len(runs))})")
    failed = [k for k, r in runs.items() if r.get("status") == "failed"]
    if failed:
        problems.append(f"{len(failed)} training run(s) failed: {failed[:5]}...")
        print("  FAILED:", failed)
else:
    print("No sweep_manifest.json — training may not have run.")

# 2) Held-out perplexities (sanity band [15, 40]).
ppls = []
for mj in sorted(models_dir.glob("ltgbert_*/metrics.json")):
    try:
        d = json.loads(mj.read_text())
        ppl = d.get("perplexity") or d.get("final_perplexity")
        if ppl is not None:
            ppls.append((mj.parent.name, float(ppl)))
    except Exception:
        pass
if ppls:
    bad = [(n, p) for n, p in ppls if not (15 <= p <= 40)]
    lo = min(p for _, p in ppls); hi = max(p for _, p in ppls)
    print(f"\\nPerplexity: {len(ppls)} models, range {lo:.1f}-{hi:.1f}.")
    if bad:
        problems.append(f"{len(bad)} model(s) out of the [15,40] perplexity band.")
        print("  OUT OF BAND:", bad[:5])

# 3) E0 (dose 0) and E_max (shared 'full' model) per construction.
eval_csv = results_dir / "eval_results.csv"
if eval_csv.exists():
    import pandas as pd
    df = pd.read_csv(eval_csv)
    self_eval = df[df["model_construction"] == df["eval_construction"]]
    e0 = (self_eval[self_eval["dose"].astype(str) == "0"]
          .groupby("eval_construction")["accuracy"].mean())
    emax = (df[df["model_construction"] == "full"]
            .groupby("eval_construction")["accuracy"].mean())
    print("\\nPer-construction E0 (zero exposure) and E_max (full corpus):")
    for c in sorted(set(e0.index) | set(emax.index)):
        print(f"  {c:<26} E0={e0.get(c, float('nan')):.3f}   "
              f"E_max={emax.get(c, float('nan')):.3f}")
    # Replication sanity: the full model on AANN should land ~[0.55, 0.75].
    aann_max = emax.get("aann")
    if aann_max is not None and not (0.55 <= aann_max <= 0.75):
        problems.append(f"AANN ceiling {aann_max:.2f} outside the ~[0.55,0.75] "
                        "replication band — check the eval pipeline.")
else:
    print("\\nNo eval_results.csv yet — eval may not have run.")

print("\\n" + "=" * 60)
if problems:
    print("RE-RUN GUIDANCE: issues found, review before trusting results:")
    for p in problems:
        print("  -", p)
else:
    print("RE-RUN GUIDANCE: no red flags. Proceed to kaggle_02_analysis.")
print("=" * 60)
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


# Comment describing the install profile for the results notebook's setup cell.
_TRAIN_COMMENT = "Training needs the ML stack: torch, transformers, accelerate, etc."


def make_run_all() -> list[dict]:
    return [
        intro_md(
            "DRC — Run the whole pipeline (single session)",
            "This is the **master** notebook: every stage end to end (data, "
            "training, evaluation, and all analysis/figures) in one go. It's "
            "right at Kaggle's ~12-hour session cap, so it suits dual-GPU or a "
            "re-run that resumes a mostly-finished pipeline. For a single-T4 run, "
            "prefer the split notebooks — `kaggle_01a_data_train` (session 1) then "
            "`kaggle_01b_eval_analysis` (session 2) — so each half fits a session.",
            "**Scope:** every stage (`only=None`).",
        ),
        # The master needs everything: ML stack + data deps + analysis deps.
        setup_code(
            extras="train",
            pip_packages="",  # deps handled by the explicit cell below
            extras_comment="Master notebook: full stack; deps installed next cell.",
        ),
        deps_code(["stanza"]),
        restore_code(),
        gpu_detect_code(),
        run_code("None  # None == run every stage", "Run all stages: only=None."),
        status_code(),
    ]


# Session 1 — the long pole: data prep through the 63-model training sweep.
_DATA_TRAIN_STAGES = [
    "download", "parse", "audit", "dose", "tokenizer", "train",
]
# Session 2 — evaluation + all analysis + figures (reads session 1's models).
_EVAL_ANALYSIS_STAGES = [
    "eval", "ngram", "hill", "model_comparison", "clustering", "transfer",
    "predictability", "generalization", "indirect_evidence", "decision", "figures",
]


def make_data_train() -> list[dict]:
    return [
        intro_md(
            "DRC 01a — Data + training (session 1, the long run)",
            "Session 1 of two. Does the expensive, hard-to-redo work: download, "
            "parse, audit, build the dose corpora, train the tokenizer, and run "
            "the **63-model pilot-gated sweep**. On one T4 (set `FORCE_SINGLE_GPU "
            "= True` below) budget ~11-12 h. Each model trains in its own "
            "subprocess and leaves a `metrics.json` when done, so if the 12 h cap "
            "cuts you off, just **re-run this notebook** — finished models are "
            "skipped and it resumes. The health-check cell at the end tells you if "
            "training looked sane. When it's green, *Save Version* and feed its "
            "output dataset to `kaggle_01b_eval_analysis`.",
            "**Scope:** `only=" + json.dumps(_DATA_TRAIN_STAGES) + "`.",
        ),
        setup_code(
            extras="train",
            pip_packages="",  # deps handled by the explicit cell below
            extras_comment=_TRAIN_COMMENT,
        ),
        deps_code(["stanza"]),
        # First notebook of the chain — usually nothing to restore, but the cell
        # lets a re-run after a partial pass merge cleanly.
        restore_code(),
        gpu_detect_code(),
        run_code(json.dumps(_DATA_TRAIN_STAGES), "Data + training (session 1)."),
        status_code(),
        results_summary_code(),
    ]


def make_eval_analysis() -> list[dict]:
    return [
        intro_md(
            "DRC 01b — Evaluation + analysis + figures (session 2)",
            "Session 2 of two. Turns the trained models into results: SLOR + "
            "n-gram evaluation, then the Hill fits, E0 indirect-evidence index, "
            "model comparison, clustering, transfer, predictability, "
            "generalization, the decision rule, and all figures. Budget ~2-3 h "
            "(eval needs the GPU; the analysis is CPU). **Attach "
            "`kaggle_01a_data_train`'s output dataset first** (*Add Input -> Your "
            "Datasets*) so the restore cell brings in the trained `models/` and "
            "the `data/`. Re-running is cheap — eval skips once "
            "`eval_results.csv` exists, so you can iterate on the analysis.",
            "**Scope:** `only=" + json.dumps(_EVAL_ANALYSIS_STAGES) + "`.",
        ),
        setup_code(
            extras="train",  # eval imports torch
            pip_packages="",  # deps handled by the explicit cell below
            extras_comment="Eval needs torch (on Kaggle already); analysis uses "
            "scipy/sklearn/matplotlib (also present).",
        ),
        deps_code(["stanza"]),
        restore_code(),
        gpu_detect_code(),
        run_code(json.dumps(_EVAL_ANALYSIS_STAGES), "Eval + analysis (session 2)."),
        status_code(),
    ]


def make_smoke() -> list[dict]:
    return [
        intro_md(
            "DRC — Smoke test (~10 min): verify the pipeline runs on Kaggle",
            "**Run this first.** It exercises the *entire* pipeline — every stage, "
            "data through figures — but on a tiny config (`configs/smoke.yaml`): a "
            "~150k-word slice of the corpus, a small 2-layer model, 1 epoch, 1 "
            "seed, and a handful of doses. It finishes in roughly **10 minutes on "
            "dual-T4** and exists to catch environment/wiring errors *before* you "
            "commit to the ~10-hour real run. It also exercises the fp16 path T4 "
            "needs. Its outputs go to **separate folders** (`data/smoke/`, "
            "`models/smoke/`, `results/smoke/`), so they never collide with or "
            "pollute the real run. If the dashboard and health check come back "
            "clean here, the full run should too.",
            "**Scope:** every stage, on the tiny `configs/smoke.yaml`.",
        ),
        setup_code(
            extras="train",
            pip_packages="",  # deps handled by the explicit cell below
            extras_comment="Smoke runs every stage; deps installed in the next cell.",
        ),
        deps_code(["stanza"]),
        restore_code(),
        gpu_detect_code(),
        run_code(
            "None  # None == every stage",
            "Run EVERY stage on the tiny smoke config.",
            config_path="configs/smoke.yaml",
        ),
        status_code(),
        results_summary_code(),
    ]


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    # Remove superseded notebooks so the directory reflects the current design.
    for stale in ("kaggle_00_data.ipynb", "kaggle_01_train.ipynb",
                  "kaggle_02_eval_analysis.ipynb", "kaggle_01_results.ipynb",
                  "kaggle_02_analysis.ipynb"):
        (NOTEBOOKS_DIR / stale).unlink(missing_ok=True)
    notebooks = {
        "kaggle_00_smoke.ipynb": make_smoke(),
        "kaggle_01a_data_train.ipynb": make_data_train(),
        "kaggle_01b_eval_analysis.ipynb": make_eval_analysis(),
        "kaggle_run_all.ipynb": make_run_all(),
    }
    for name, cells in notebooks.items():
        write_notebook(NOTEBOOKS_DIR / name, cells)
    print(f"Generated {len(notebooks)} notebooks in {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
