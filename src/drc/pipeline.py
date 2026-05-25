"""Fault-tolerant pipeline runner for the DRC stages.

The pipeline is genuinely sequential — you can't train without corpora or fit
curves without eval results — but a long Kaggle session shouldn't fall over just
because one stage hit a bad batch or an OOM. So this runner gives us three things
the bare CLIs don't:

1. **Crash isolation.** Every stage runs in its own subprocess. A segfault, a
   CUDA OOM, or a hard kill takes down that subprocess and nothing else — the
   notebook kernel survives, the runner records the failure, and it keeps going.
2. **Idempotent resume.** Each stage knows what it produces. Re-run the pipeline
   after a timeout and finished stages are skipped, so you pick up where you
   stopped instead of redoing seven hours of training.
3. **Honest dependency gating.** A stage whose inputs never arrived is marked
   ``blocked`` rather than run-and-crash. Independent stages still run — the
   n-gram baseline only needs the corpora, so it goes ahead even if the whole
   training sweep failed.

Nothing here fabricates output. A blocked or failed stage produces no results;
it just says so clearly and lets the rest of the pipeline proceed.

CLI::

    python -m drc.pipeline --config configs/base.yaml
    python -m drc.pipeline --config configs/base.yaml --single-gpu
    python -m drc.pipeline --config configs/base.yaml --only ngram,hill,figures
    python -m drc.pipeline --config configs/base.yaml --force train
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Stage outcomes. "done" and "skipped" both count as satisfied when a downstream
# stage checks its dependencies; "failed" and "blocked" do not.
DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"
BLOCKED = "blocked"

# How many lines of a failed subprocess's output to keep in the status file.
_ERROR_TAIL_LINES = 40


@dataclass
class StageResult:
    name: str
    status: str
    seconds: float = 0.0
    detail: str = ""

    @property
    def satisfied(self) -> bool:
        """Did this stage leave its outputs in place for dependents to use?"""
        return self.status in (DONE, SKIPPED)


@dataclass
class Stage:
    """One unit of pipeline work.

    ``action`` does the work and raises on failure — that's the whole contract.
    ``is_done`` lets a stage declare itself already complete (used for resume and
    idempotency); when it returns True the action is skipped. ``requires`` names
    the stages that must be satisfied first. A non-critical stage that fails does
    not abort the run; a critical one does.
    """

    name: str
    action: Callable[[], None]
    is_done: Callable[[], bool] = lambda: False
    requires: Sequence[str] = field(default_factory=tuple)
    critical: bool = False
    description: str = ""


def shell_action(command: Sequence[str], cwd: Path | None = None) -> Callable[[], None]:
    """Wrap a subprocess command as a Stage action.

    Streams the child's output live (so you watch progress in the notebook) while
    also keeping the tail, and raises ``RuntimeError`` with that tail if the
    command exits non-zero. Running out-of-process is the point: a crash in the
    child can't take the parent down with it.
    """

    def _run() -> None:
        logger.info("$ %s", " ".join(str(c) for c in command))
        proc = subprocess.Popen(
            [str(c) for c in command],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line)
            if len(tail) > _ERROR_TAIL_LINES:
                tail.pop(0)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(
                f"exit code {code}\n--- last {len(tail)} lines ---\n{''.join(tail)}"
            )

    return _run


def _all_exist(paths: Iterable[Path]) -> bool:
    paths = list(paths)
    return bool(paths) and all(p.exists() for p in paths)


def run_pipeline(
    stages: Sequence[Stage],
    status_path: Path | None = None,
    force: Iterable[str] = (),
    only: Iterable[str] | None = None,
) -> dict[str, StageResult]:
    """Run the stages in order, isolating failures and persisting status.

    Stages are expected in dependency order (each stage's ``requires`` point only
    at earlier ones). ``force`` re-runs stages even if they look done; ``only``
    restricts the run to a subset (their dependencies must already be satisfied).
    Returns the result map and, if ``status_path`` is given, writes it as JSON
    after every stage so a killed session still leaves a readable record.
    """
    force = set(force)
    only_set = set(only) if only is not None else None
    results: dict[str, StageResult] = {}

    def persist() -> None:
        if status_path is not None:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {name: asdict(r) for name, r in results.items()}
            status_path.write_text(json.dumps(payload, indent=2))

    for stage in stages:
        if only_set is not None and stage.name not in only_set:
            continue

        # Already finished? Skip it — this is what makes a re-run resume cleanly.
        if stage.name not in force:
            try:
                already = stage.is_done()
            except Exception:  # a flaky completion check shouldn't abort the run
                already = False
            if already:
                results[stage.name] = StageResult(stage.name, SKIPPED, detail="already complete")
                logger.info("[skip] %s — already complete", stage.name)
                persist()
                continue

        # Any unmet dependency? Block rather than run-and-crash.
        unmet = [
            dep for dep in stage.requires
            if dep not in results or not results[dep].satisfied
        ]
        if unmet:
            results[stage.name] = StageResult(
                stage.name, BLOCKED, detail=f"waiting on: {', '.join(unmet)}"
            )
            logger.warning("[block] %s — unmet dependencies: %s", stage.name, ", ".join(unmet))
            persist()
            continue

        logger.info("[run] %s — %s", stage.name, stage.description or "")
        t0 = time.time()
        try:
            stage.action()
            results[stage.name] = StageResult(stage.name, DONE, seconds=time.time() - t0)
            logger.info("[done] %s (%.1fs)", stage.name, time.time() - t0)
        except Exception as exc:  # noqa: BLE001 — isolation is the whole point
            results[stage.name] = StageResult(
                stage.name, FAILED, seconds=time.time() - t0, detail=str(exc)
            )
            tag = "critical " if stage.critical else ""
            # We never abort the whole run on a failure — not even a critical one.
            # A failed stage just means its dependents get blocked; everything
            # independent still runs. The dashboard shows exactly what happened.
            logger.error(
                "[fail] %s%s — %s; continuing (dependents will be blocked).",
                tag, stage.name, exc,
            )
            persist()
            continue
        persist()

    print_dashboard(results)
    return results


def print_dashboard(results: dict[str, StageResult]) -> None:
    """Print a compact end-of-run summary table."""
    icon = {DONE: "✓", SKIPPED: "·", FAILED: "✗", BLOCKED: "∅"}
    width = max((len(n) for n in results), default=4)
    lines = ["", "Pipeline summary", "-" * (width + 24)]
    for r in results.values():
        t = f"{r.seconds:6.1f}s" if r.seconds else "       "
        note = f"  {r.detail}" if r.detail and r.status != DONE else ""
        lines.append(f"  {icon.get(r.status, '?')} {r.name:<{width}}  {r.status:<7} {t}{note}")
    done = sum(r.status == DONE for r in results.values())
    failed = [n for n, r in results.items() if r.status == FAILED]
    blocked = [n for n, r in results.items() if r.status == BLOCKED]
    lines.append("-" * (width + 24))
    lines.append(f"  {done} ran, {len(failed)} failed, {len(blocked)} blocked")
    if failed:
        lines.append(f"  failed:  {', '.join(failed)}")
    if blocked:
        lines.append(f"  blocked: {', '.join(blocked)}")
    print("\n".join(lines))


def default_phases(config_path: Path, single_gpu: bool = False) -> list[Stage]:
    """The standard DRC pipeline, wired with dependencies and completion checks.

    Each stage shells out to a module CLI, so a crash stays contained. The
    completion checks double as resume markers. Note the deliberate independence:
    ``ngram`` hangs off the corpora, not the models, so it survives a failed
    training sweep; ``audit`` is a quality gate that never blocks the data path.
    """
    from drc import CONSTRUCTIONS
    from drc.data.download import load_config, resolve_path

    cfg = load_config(config_path)
    paths = cfg["paths"]

    def p(key: str) -> Path:
        return resolve_path(config_path, paths[key])

    results_dir = p("results")
    dose_dir = p("dose_corpora")
    n_dose_corpora = len(CONSTRUCTIONS) * 5  # 4 constructions x 5 dose levels

    def cli(module: str, *extra: str) -> Callable[[], None]:
        return shell_action([sys.executable, "-m", module, "--config", str(config_path), *extra])

    def manifest_complete() -> bool:
        manifest = p("models") / "sweep_manifest.json"
        if not manifest.exists():
            return False
        try:
            data = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        runs = data.get("runs", data) if isinstance(data, dict) else data
        statuses = [r.get("status") for r in runs.values()] if isinstance(runs, dict) else []
        return bool(statuses) and all(s in ("done", "complete", "ok") for s in statuses)

    sweep_extra = ("--single-gpu",) if single_gpu else ()

    return [
        Stage(
            "download", cli("drc.data.download"),
            # raw_corpus is a single text file (domain<TAB>text per line); treat a
            # non-empty file as done.
            is_done=lambda: p("raw_corpus").is_file() and p("raw_corpus").stat().st_size > 0,
            critical=True, description="fetch BabyLM-10M + replacement pool",
        ),
        Stage(
            "parse", cli("drc.data.parse"),
            # Parse produces both the corpus and the pool CoNLL-U; dose needs both.
            is_done=lambda: p("parsed").exists() and p("parsed_pool").exists(),
            requires=("download",), critical=True, description="Stanza parse (corpus + pool)",
        ),
        Stage(
            "audit", cli("drc.data.qa_audit", "--judge", "manual"),
            is_done=lambda: p("filtered").exists() and any(p("filtered").glob("*positives*")),
            requires=("parse",), description="filter precision/recall gate (non-blocking)",
        ),
        Stage(
            "dose", cli("drc.data.dose_corpora"),
            is_done=lambda: len(list(dose_dir.glob("*_dose-*.conllu"))) >= n_dose_corpora,
            requires=("parse",), critical=True, description="build the 20 dose corpora",
        ),
        Stage(
            "tokenizer", cli("drc.tokenizer.train_tokenizer"),
            is_done=lambda: (p("tokenizer") / "tokenizer.json").exists(),
            requires=("dose",), description="shared 16k BPE tokenizer",
        ),
        Stage(
            "train", cli("drc.train.sweep", *sweep_extra),
            is_done=manifest_complete,
            requires=("tokenizer",), description="pilot-gated 60-run sweep (dual-T4)",
        ),
        Stage(
            "eval", cli("drc.eval.run_eval"),
            is_done=lambda: (results_dir / "eval_results.csv").exists(),
            requires=("train",), description="SLOR acceptability on all four test sets",
        ),
        Stage(
            "ngram", cli("drc.eval.ngram_baseline"),
            is_done=lambda: (results_dir / "ngram_baseline.csv").exists(),
            requires=("dose",), description="4-gram baseline (independent of training)",
        ),
        Stage(
            "hill", cli("drc.analysis.hill"),
            is_done=lambda: (results_dir / "hill_fits.csv").exists(),
            requires=("eval",), description="Hill dose-response fits + bootstrap CIs",
        ),
        Stage(
            "model_comparison", cli("drc.analysis.model_comparison"),
            is_done=lambda: (results_dir / "model_comparison.csv").exists(),
            requires=("eval",), description="Hill vs power/step/log-linear/null",
        ),
        Stage(
            "clustering", cli("drc.analysis.clustering"),
            is_done=lambda: (results_dir / "cluster_assignments.csv").exists(),
            requires=("hill",), description="k-means over fitted parameters",
        ),
        Stage(
            "decision", cli("drc.analysis.decision"),
            is_done=lambda: (results_dir / "decision.txt").exists(),
            requires=("hill",), description="fire the pre-registered decision rule",
        ),
        Stage(
            "figures", cli("drc.analysis.figures"),
            is_done=lambda: len(list((results_dir / "figures").glob("fig*.pdf"))) >= 5,
            requires=("hill",), description="render the core paper figures",
        ),
        # The three analyses below widen the contribution beyond the closest
        # prior work (Oba et al. 2024), and need no new training — they reuse the
        # eval outputs and fitted curves.
        Stage(
            "transfer", cli("drc.analysis.transfer"),
            is_done=lambda: (results_dir / "transfer_matrix.csv").exists(),
            requires=("eval",), description="cross-construction transfer dose-response",
        ),
        Stage(
            "predictability", cli("drc.analysis.predictability"),
            is_done=lambda: (results_dir / "predictability.csv").exists(),
            requires=("hill",), description="predict E50 from corpus properties (RQ4)",
        ),
        Stage(
            "generalization", cli("drc.analysis.generalization"),
            is_done=lambda: (results_dir / "generalization.csv").exists(),
            requires=("eval",), description="memorization vs generalization (seen/novel fillers)",
        ),
        Stage(
            "indirect_evidence", cli("drc.analysis.indirect_evidence"),
            is_done=lambda: (results_dir / "indirect_evidence.csv").exists(),
            requires=("eval",), description="E0 indirect-evidence index (headline) + fig9",
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--single-gpu", action="store_true", help="single-T4 fallback (48 runs)")
    parser.add_argument("--only", default=None, help="comma-separated subset of stages to run")
    parser.add_argument("--force", default="", help="comma-separated stages to re-run even if done")
    parser.add_argument("--status", type=Path, default=None, help="where to write the status JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    status = args.status or (args.config.parent.parent / "results" / "pipeline_status.json")
    stages = default_phases(args.config, single_gpu=args.single_gpu)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    force = [s.strip() for s in args.force.split(",") if s.strip()]

    results = run_pipeline(stages, status_path=status, force=force, only=only)
    # Exit non-zero only if something genuinely failed (blocked-by-design is fine).
    return 1 if any(r.status == FAILED for r in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
