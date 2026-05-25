"""Run the whole grid: every construction x dose x seed.

That's 4 x 5 x 3 = 60 models. This module is the conductor, not the orchestra —
it never trains anything itself. Each run is launched as a fresh
``python -m drc.train.train`` subprocess, which buys two things: a crash or a NaN
abort in one run can't take down the sweep, and the per-run determinism in
``train.py`` isn't disturbed by state left over from a previous run in the same
process.

A few habits make a long sweep survivable on Kaggle's session limits:

* **Pilot first.** Before committing to 60 runs we train the canonical
  ``aann / all / seed 42`` model and check its held-out perplexity against the
  config's ``pilot_max_perplexity``. If the pilot's off, something's wrong with
  the data or the recipe and we refuse to burn hours on the rest.
* **Resume by skipping.** A finished run leaves a ``metrics.json``. We skip any
  run whose metrics already exist, so re-launching after a timeout picks up
  where it left off.
* **Two GPUs when we have them.** Kaggle's dual-T4 lets us run two workers in
  parallel, pinned with ``CUDA_VISIBLE_DEVICES``. ``--single-gpu`` falls back to
  one worker and, per the PRD degradation plan, drops ``dose=4`` to bring the
  grid down to 48 runs that fit in the session budget.

A manifest JSON tracks each run's status and is rewritten after every state
change, so you can always see what's done, running, or failed.

CLI::

    python -m drc.train.sweep --config configs/base.yaml
    python -m drc.train.sweep --config configs/base.yaml --single-gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from drc.data.download import load_config, resolve_path

from .train import run_name

logger = logging.getLogger(__name__)

# The canonical pilot run. If this one's healthy, the recipe is sound.
PILOT_CONSTRUCTION = "aann"
PILOT_DOSE = "all"
PILOT_SEED = 42

# Single-GPU mode used to drop a dose to save time, but that left core
# constructions with only 4 dose points (an exactly-determined Hill fit with no
# room for a meaningful CI). We'd rather keep all 5 points and save time via
# fewer epochs instead, so single-GPU now runs the full grid sequentially.
SINGLE_GPU_DROP_DOSE = None


def _grid(config: dict[str, Any], drop_dose: Any | None = None) -> list[tuple[str, Any, int]]:
    """The tiered set of runs from the design block (not a full factorial).

    See ``drc.design``: every construction at dose 0, the full ladder for the
    core subset, and one shared full-corpus model per seed. ``drop_dose`` trims
    the core ladder for the single-GPU path.
    """
    from drc.design import run_cells

    return run_cells(config, drop_dose=drop_dose)


def _metrics_path(config: dict[str, Any], config_path: Path, run: tuple[str, Any, int]) -> Path:
    construction, dose, seed = run
    models_root = resolve_path(config_path, config["paths"]["models"])
    return models_root / run_name(construction, dose, seed) / "metrics.json"


def is_done(config: dict[str, Any], config_path: Path, run: tuple[str, Any, int]) -> bool:
    """A run is done when it left a readable ``metrics.json``."""
    path = _metrics_path(config, config_path, run)
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            json.load(fh)
        return True
    except (json.JSONDecodeError, OSError):
        # A truncated metrics file means the run died mid-write — redo it.
        return False


def _launch(
    config_path: Path, run: tuple[str, Any, int], gpu_id: int | None
) -> subprocess.Popen:
    """Start one training run as a subprocess, optionally pinned to a GPU."""
    import os

    construction, dose, seed = run
    cmd = [
        sys.executable, "-m", "drc.train.train",
        "--construction", construction,
        "--dose", str(dose),
        "--seed", str(seed),
        "--config", str(config_path),
    ]
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(
        "Launching %s (gpu=%s).", run_name(construction, dose, seed), gpu_id,
    )
    return subprocess.Popen(cmd, env=env)


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Rewrite the manifest atomically-ish so a kill mid-write can't corrupt it."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    tmp.replace(manifest_path)


def _pilot_run(config: dict[str, Any]) -> tuple[str, Any, int]:
    """Pick the pilot from the design.

    With a shared full-corpus ceiling, that full model is the natural pilot: it
    trains on the unfiltered corpus, it's always in the run set, and a healthy
    perplexity there means the recipe is sound. Otherwise fall back to the first
    core construction at dose 'all' (the legacy pilot).
    """
    from drc.design import FULL_CORPUS_CODE, core_constructions, seeds, shares_full_corpus

    s = (seeds(config) or [PILOT_SEED])[0]
    if shares_full_corpus(config):
        return (FULL_CORPUS_CODE, "all", s)
    core = core_constructions(config)
    return (core[0] if core else PILOT_CONSTRUCTION, PILOT_DOSE, s)


def run_pilot(config_path: Path, config: dict[str, Any]) -> bool:
    """Train the pilot and check its perplexity against the config gate.

    Returns ``True`` if the pilot passes (or was already done and passing).
    Trains in-process — there's only one, and we want its exception, if any, to
    surface here rather than in a detached subprocess.
    """
    from .train import train_one

    pilot = _pilot_run(config)
    threshold = float(config["evaluation"]["pilot_max_perplexity"])

    if is_done(config, config_path, pilot):
        logger.info("Pilot already trained; reading its metrics.")
    else:
        logger.info("Training pilot %s before the full sweep...", run_name(*pilot))
        train_one(config_path, *pilot)

    with open(_metrics_path(config, config_path, pilot), encoding="utf-8") as fh:
        metrics = json.load(fh)
    ppl = metrics.get("heldout_perplexity", float("inf"))

    if ppl is None or ppl != ppl:  # None or NaN
        logger.error("Pilot perplexity is %s — recipe is broken. Stopping.", ppl)
        return False
    if ppl >= threshold:
        logger.error(
            "Pilot perplexity %.3f >= gate %.3f. Refusing to launch the sweep; "
            "fix the data or recipe first.", ppl, threshold,
        )
        return False
    logger.info("Pilot OK: perplexity %.3f < gate %.3f.", ppl, threshold)
    return True


def _run_sequential(
    config_path: Path, config: dict[str, Any], pending: list, manifest: dict, manifest_path: Path
) -> None:
    """One run at a time on a single GPU."""
    for run in pending:
        key = run_name(*run)
        manifest["runs"][key]["status"] = "running"
        _write_manifest(manifest_path, manifest)

        proc = _launch(config_path, run, gpu_id=0)
        ret = proc.wait()

        ok = ret == 0 and is_done(config, config_path, run)
        manifest["runs"][key]["status"] = "done" if ok else "failed"
        manifest["runs"][key]["returncode"] = ret
        _write_manifest(manifest_path, manifest)
        if not ok:
            logger.error("Run %s failed (rc=%s); continuing with the rest.", key, ret)


def _run_parallel(
    config_path: Path, config: dict[str, Any], pending: list, manifest: dict,
    manifest_path: Path, n_gpus: int = 2, poll_seconds: float = 5.0,
) -> None:
    """Keep ``n_gpus`` runs in flight at once, each pinned to its own GPU."""
    queue = list(pending)
    active: dict[int, tuple] = {}  # gpu_id -> (run, Popen)

    def _start_on(gpu_id: int) -> None:
        if not queue:
            return
        run = queue.pop(0)
        key = run_name(*run)
        manifest["runs"][key]["status"] = "running"
        manifest["runs"][key]["gpu"] = gpu_id
        _write_manifest(manifest_path, manifest)
        active[gpu_id] = (run, _launch(config_path, run, gpu_id=gpu_id))

    for gpu_id in range(n_gpus):
        _start_on(gpu_id)

    while active:
        time.sleep(poll_seconds)
        for gpu_id in list(active.keys()):
            run, proc = active[gpu_id]
            ret = proc.poll()
            if ret is None:
                continue
            key = run_name(*run)
            ok = ret == 0 and is_done(config, config_path, run)
            manifest["runs"][key]["status"] = "done" if ok else "failed"
            manifest["runs"][key]["returncode"] = ret
            _write_manifest(manifest_path, manifest)
            if not ok:
                logger.error("Run %s failed (rc=%s); continuing.", key, ret)
            del active[gpu_id]
            _start_on(gpu_id)  # backfill the freed GPU


def run_sweep(
    config_path: Path,
    single_gpu: bool = False,
    skip_pilot: bool = False,
) -> None:
    """Run the pilot gate, then launch the rest of the grid with resume + manifest."""
    config = load_config(config_path)

    if not skip_pilot:
        if not run_pilot(config_path, config):
            raise SystemExit("Pilot gate failed; sweep not launched.")

    drop = SINGLE_GPU_DROP_DOSE if single_gpu else None
    if single_gpu:
        logger.info("Single-GPU mode: running the full grid sequentially.")
    grid = _grid(config, drop_dose=drop)

    results_root = resolve_path(config_path, config["paths"]["results"])
    manifest_path = results_root / "sweep_manifest.json"
    manifest: dict[str, Any] = {
        "mode": "single-gpu" if single_gpu else "dual-gpu",
        "total_runs": len(grid),
        "runs": {},
    }

    pending: list = []
    for run in grid:
        key = run_name(*run)
        if is_done(config, config_path, run):
            manifest["runs"][key] = {"status": "done", "skipped": True}
        else:
            manifest["runs"][key] = {"status": "pending"}
            pending.append(run)
    _write_manifest(manifest_path, manifest)

    logger.info(
        "Sweep: %d total, %d already done, %d to run.",
        len(grid), len(grid) - len(pending), len(pending),
    )
    if not pending:
        logger.info("Nothing to do — every run is already complete.")
        return

    if single_gpu:
        _run_sequential(config_path, config, pending, manifest, manifest_path)
    else:
        _run_parallel(config_path, config, pending, manifest, manifest_path)

    n_done = sum(1 for r in manifest["runs"].values() if r["status"] == "done")
    n_failed = sum(1 for r in manifest["runs"].values() if r["status"] == "failed")
    logger.info("Sweep finished: %d done, %d failed. Manifest: %s",
                n_done, n_failed, manifest_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base training config.",
    )
    parser.add_argument(
        "--single-gpu", action="store_true",
        help="Run sequentially on one GPU and drop dose=4 (48 runs).",
    )
    parser.add_argument(
        "--skip-pilot", action="store_true",
        help="Skip the pilot gate (use only when the pilot already passed).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run_sweep(args.config, single_gpu=args.single_gpu, skip_pilot=args.skip_pilot)


if __name__ == "__main__":
    main()
