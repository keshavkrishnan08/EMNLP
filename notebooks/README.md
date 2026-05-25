# Kaggle notebooks (dual-T4)

These four notebooks run the DRC pipeline on Kaggle. They're thin wrappers over
the fault-tolerant runner in [`src/drc/pipeline.py`](../src/drc/pipeline.py) —
they make the `drc` package importable, restore any prior artifacts, detect the
GPUs, and call `run_pipeline`. They don't reinvent orchestration.

All four are **generated** by [`_build.py`](_build.py). That script is the single
source of truth. Don't hand-edit the `.ipynb` files — change the cell text in
`_build.py` and regenerate:

```bash
python notebooks/_build.py
```

## The four notebooks

| Notebook | Stages it runs | When to use it |
|----------|----------------|----------------|
| `kaggle_run_all.ipynb` | all 13 (`only=None`) | One-shot, single session. Re-run to resume after a timeout. |
| `kaggle_00_data.ipynb` | download, parse, audit, dose, tokenizer | First link in the chain. Builds the corpora + tokenizer. |
| `kaggle_01_train.ipynb` | train | The long dual-T4 sweep. Resumable; chains after 00. |
| `kaggle_02_eval_analysis.ipynb` | eval, ngram, hill, model_comparison, clustering, decision, figures | Turns models into results, curves, and figures. Chains after 01. |

The 13 stages, in order: `download, parse, audit, dose, tokenizer, train, eval,
ngram, hill, model_comparison, clustering, decision, figures`.

## Accelerator setting

Set **Settings -> Accelerator -> GPU T4 x2**. With two cards the training sweep
runs two jobs in parallel (one pinned per GPU); with one it falls back to a
48-run single-GPU plan (it drops `dose=4`). The GPU-detect cell figures out
which mode you're in and sets `SINGLE_GPU` for you.

## The bf16 -> fp16 caveat for T4

T4 is a Turing-generation card and **does not support bf16**. The shipped
[`configs/base.yaml`](../configs/base.yaml) sets `precision: bf16`. Before you
train on T4, edit that line to:

```yaml
  precision: fp16
```

The notebooks **warn** you about this in the GPU-detect cell but do not edit the
config automatically — changing a frozen training config is your call, not the
notebook's.

## Chaining via notebook-output datasets

The pipeline is split so a long run survives Kaggle's 12-hour session limit and
each phase can be re-run on its own. The notebooks pass state through Kaggle's
**notebook-output datasets**:

1. Run `kaggle_00_data` to completion, then **Save Version**. Kaggle snapshots
   `/kaggle/working` as a notebook-output dataset.
2. Open `kaggle_01_train` and **Add Input -> Your Datasets**, picking that
   output. It mounts read-only under `/kaggle/input/<dataset-name>/`.
3. The **restore-prior-artifacts** cell copies any `data/`, `models/`, and
   `results/` it finds under `/kaggle/input/*/` into the working repo. The
   already-done stages then detect their outputs and resume as *skipped*, so the
   notebook only does its own part.
4. Repeat: `kaggle_01_train`'s saved output becomes the input to
   `kaggle_02_eval_analysis`.

`kaggle_run_all` doesn't need chaining — it does everything in one session — but
it still restores prior artifacts, so re-running it after a timeout resumes
cleanly too.

## Resume after a timeout

Re-run the notebook. That's it. Every stage runs in its own subprocess and
records its outputs; `run_pipeline` skips any stage that's already complete, so
a re-run picks up where the session was cut off. A kernel crash *during* the
training sweep can't kill the notebook either — the crashing run is its own
subprocess, and finished runs leave a `metrics.json` that marks them done.

## Nothing is fabricated

A blocked or failed stage produces **no numbers** — it just says so in the
dashboard and lets the rest of the pipeline proceed where it can. If training
never ran, the `ngram` baseline (which only needs the corpora) still runs, but
the model-dependent results simply won't exist. The notebooks never invent
results to fill a gap.

## Repo setup inside the notebook

The setup cell needs your repo's clone URL. Open the notebook and edit:

```python
REPO_URL = "https://github.com/your-org/drc-emnlp-2026.git"  # <-- EDIT ME
```

If you instead upload the repo as a Kaggle dataset, the setup cell finds it
under `/kaggle/input/*/` and skips the clone. Either way it installs the package
editable (`pip install -e .[train]`), degrading to a `sys.path` fallback if the
install fails.
