"""Score every trained model on every construction's minimal pairs.

This is the main evaluation driver. For each of the 60 runs (4 constructions x
5 doses x 3 seeds) we load the checkpoint and score it against *all four*
construction test sets — not just its own. Cross-construction scoring is the
point: if dosing a model on AANN also moves its tough-movement accuracy, that's
a collateral effect we want to see, not hide.

Per minimal pair the rule is simple: the model is "correct" when SLOR(good) >
SLOR(bad). We report per-(model, eval-construction) mean accuracy and the
binomial standard error.

Results stream to ``results/eval_results.csv`` one row at a time, so a crash
midway through a long sweep loses at most the row in flight. Re-running skips
any (model, eval-construction) pair already in the CSV, so it resumes cleanly.

The unigram correction inside SLOR is built from each model's *own* dose corpus
— the corpus it actually trained on — so the normaliser matches the model's
word-frequency world.

CLI::

    python -m drc.eval.run_eval --config configs/base.yaml
    python -m drc.eval.run_eval --config configs/base.yaml --construction aann
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, NamedTuple

from drc import CONSTRUCTIONS, DOSE_LEVELS, SEEDS
from drc.data.download import load_config, resolve_path

logger = logging.getLogger(__name__)

# Columns in results/eval_results.csv. Kept as a constant so the writer, the
# resume logic, and the sanity check all agree on the schema.
CSV_FIELDS = (
    "model_construction",
    "dose",
    "seed",
    "eval_construction",
    "n_pairs",
    "n_correct",
    "accuracy",
    "std_error",
)

# Columns in results/eval_per_item.csv. This is the additive, per-minimal-pair
# log that the memorization-vs-generalization analysis reads. It records the two
# raw SLOR scores so a downstream analysis can re-derive correctness or do
# anything else with the margins — the aggregate CSV throws those away.
#
# The trailing three columns are a *second* acceptability measure, the plain
# length-normalised pseudo-log-likelihood (mean-LP) with no unigram correction.
# It rides along for free: it's already computed inside SLOR, so recording it
# costs nothing and lets the measure-robustness check show the dose-response
# doesn't hinge on the SLOR metric specifically. ``correct_meanlp`` mirrors
# ``correct`` but scores the pair on mean-LP instead.
PER_ITEM_FIELDS = (
    "model_construction",
    "dose",
    "seed",
    "eval_construction",
    "item_id",
    "slor_good",
    "slor_bad",
    "correct",
    "meanlp_good",
    "meanlp_bad",
    "correct_meanlp",
)

# Pilot sanity band: the AANN model trained on the full dose at seed 42 should
# replicate Misra & Mahowald (2024) on AANN items. Outside this and something's
# off — we warn rather than crash, since a single odd run shouldn't kill a sweep.
SANITY_MODEL = ("aann", "all", 42)
SANITY_CONSTRUCTION = "aann"
SANITY_ACC_RANGE = (0.55, 0.75)


class EvalItem(NamedTuple):
    """One minimal pair read from an eval_items JSONL file."""

    item_id: str
    construction: str
    good_sentence: str
    bad_sentence: str


class ModelRun(NamedTuple):
    """A discovered checkpoint and the (construction, dose, seed) it belongs to."""

    construction: str
    dose: str
    seed: int
    path: Path


def _run_name(construction: str, dose: Any, seed: int) -> str:
    """Run name for a (construction, dose, seed), via the trainer when available.

    The trainer owns the canonical naming. We import it lazily so eval doesn't
    hard-depend on train (and so this module imports in a bare environment). If
    it isn't written yet, fall back to the obvious ``<c>_dose-<d>_seed-<s>``.
    """
    try:
        from drc.train.train import run_name

        return run_name(construction, dose, seed)
    except Exception:  # noqa: BLE001 - train may not exist yet; fall back
        return f"{construction}_dose-{dose}_seed-{seed}"


# Pull (construction, dose, seed) back out of a run directory name. We match the
# pieces by the known vocabulary rather than positionally, so a tweak to the
# trainer's separator style won't silently break discovery.
_DOSE_ALTERNATION = "|".join(re.escape(str(d)) for d in DOSE_LEVELS)
_SEED_ALTERNATION = "|".join(str(s) for s in SEEDS)
_CONS_ALTERNATION = "|".join(re.escape(c) for c in CONSTRUCTIONS)
# Tolerant of both the trainer's "ltgbert_<c>_D<dose>_seed<seed>" style and a
# plainer "<c>_dose-<dose>_seed-<seed>": the dose marker is "dose-", "dose_",
# "dose", or a bare "D", and likewise for seed.
_RUN_RE = re.compile(
    rf"(?P<construction>{_CONS_ALTERNATION}).*?"
    rf"(?:dose[-_]?|D)(?P<dose>{_DOSE_ALTERNATION}).*?"
    rf"seed[-_]?(?P<seed>{_SEED_ALTERNATION})"
)


def _looks_like_checkpoint(path: Path) -> bool:
    """A directory holding HF weights — config plus a weights file."""
    if not (path / "config.json").exists():
        return False
    return any(
        (path / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def discover_models(models_root: Path) -> list[ModelRun]:
    """Find trained checkpoints under ``models/`` and tag each with its run id.

    We try the trainer's exact run name first (matching all 60 expected dirs),
    then fall back to regex-parsing any other directory that looks like a
    checkpoint. That second pass catches runs whose naming drifted, so they
    still get evaluated instead of silently dropped.
    """
    models_root = Path(models_root)
    if not models_root.exists():
        logger.warning("Models directory %s doesn't exist yet.", models_root)
        return []

    found: dict[Path, ModelRun] = {}

    # First pass: the expected 60 by canonical name.
    for construction in CONSTRUCTIONS:
        for dose in DOSE_LEVELS:
            for seed in SEEDS:
                name = _run_name(construction, str(dose), seed)
                path = models_root / name
                if _looks_like_checkpoint(path):
                    found[path.resolve()] = ModelRun(
                        construction, str(dose), seed, path
                    )

    # Second pass: anything else that parses and carries weights.
    for path in sorted(models_root.iterdir()):
        if not path.is_dir() or path.resolve() in found:
            continue
        if not _looks_like_checkpoint(path):
            continue
        m = _RUN_RE.search(path.name)
        if m:
            found[path.resolve()] = ModelRun(
                m.group("construction"), m.group("dose"),
                int(m.group("seed")), path,
            )
        else:
            logger.warning("Skipping unparseable checkpoint dir: %s", path.name)

    runs = sorted(found.values(), key=lambda r: (r.construction, r.dose, r.seed))
    logger.info("Discovered %d trained models under %s.", len(runs), models_root)
    return runs


def read_eval_items(path: Path) -> list[EvalItem]:
    """Load one construction's minimal pairs from its JSONL file."""
    items: list[EvalItem] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                items.append(
                    EvalItem(
                        item_id=str(obj["item_id"]),
                        construction=str(obj["construction"]),
                        good_sentence=obj["good_sentence"],
                        bad_sentence=obj["bad_sentence"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path.name} line {lineno} missing field {exc}"
                ) from exc
    return items


def _binomial_std_error(accuracy: float, n: int) -> float:
    """Standard error of a proportion: sqrt(p(1-p)/n). Zero when n is zero."""
    if n == 0:
        return 0.0
    return math.sqrt(accuracy * (1.0 - accuracy) / n)


def _load_model_and_tokenizer(run: ModelRun, tokenizer_path: Path):
    """Load a checkpoint and its tokenizer wrapper, deferring to the trainer.

    Imports are lazy on purpose: torch/transformers and the teammate's train
    module only get touched when we actually evaluate. We prefer the trainer's
    own builders so eval loads weights exactly the way training saved them.
    """
    import torch

    from drc.train.model import build_tokenizer_wrapper

    tokenizer = build_tokenizer_wrapper(str(tokenizer_path))

    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(str(run.path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _load_existing_keys(csv_path: Path) -> set[tuple[str, str, str, str]]:
    """Keys already scored, so a resumed run skips them.

    The key is (model_construction, dose, seed, eval_construction) as strings —
    everything that uniquely identifies a row.
    """
    done: set[tuple[str, str, str, str]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            done.add(
                (row["model_construction"], row["dose"],
                 row["seed"], row["eval_construction"])
            )
    logger.info("Resuming: %d (model, construction) rows already in %s.",
                len(done), csv_path.name)
    return done


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one result row, writing the header first if the file is new."""
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _append_per_item_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    """Append a batch of per-item rows, writing the header first if file is new.

    We write a whole (model, eval-construction) block at once rather than row by
    row. That keeps the per-item file's resume granularity identical to the
    aggregate's — a block is present in both files or in neither — so the two
    never drift out of sync after a crash. The header is the PER_ITEM_FIELDS
    constant so the writer and any reader agree on column order.
    """
    if not rows:
        return
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_ITEM_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _per_item_rows(
    run_info: ModelRun,
    eval_construction: str,
    scored: list[tuple],
) -> list[dict[str, Any]]:
    """Turn scored minimal pairs into per-item CSV rows.

    Pulled out as its own pure function so a test can exercise the row shape
    without torch in the room — feed it mock scores and check the columns. Each
    tuple is ``(item_id, slor_good, slor_bad)`` or, when the mean-LP measure is
    captured, ``(item_id, slor_good, slor_bad, meanlp_good, meanlp_bad)``. We
    accept both so the older three-element form keeps working; missing mean-LP
    values come out blank rather than a fabricated zero.

    ``correct`` mirrors the aggregate rule exactly: strictly good > bad on SLOR.
    ``correct_meanlp`` is the same rule on the second measure, left blank when
    no mean-LP scores were supplied.
    """
    rows: list[dict[str, Any]] = []
    for entry in scored:
        item_id, good, bad = entry[0], entry[1], entry[2]
        has_meanlp = len(entry) >= 5
        ml_good = float(entry[3]) if has_meanlp else None
        ml_bad = float(entry[4]) if has_meanlp else None
        rows.append(
            {
                "model_construction": run_info.construction,
                "dose": run_info.dose,
                "seed": run_info.seed,
                "eval_construction": eval_construction,
                "item_id": item_id,
                "slor_good": round(float(good), 6),
                "slor_bad": round(float(bad), 6),
                "correct": int(good > bad),
                "meanlp_good": round(ml_good, 6) if has_meanlp else "",
                "meanlp_bad": round(ml_bad, 6) if has_meanlp else "",
                "correct_meanlp": int(ml_good > ml_bad) if has_meanlp else "",
            }
        )
    return rows


def evaluate_pair_set(
    model: Any,
    tokenizer: Any,
    device: str,
    items: list[EvalItem],
    unigram_counts,
    *,
    mask_batch_size: int,
) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Score one model on one construction's pairs under both measures.

    Returns ``(n_pairs, n_correct, scored)`` where ``scored`` is one
    ``(item_id, slor_good, slor_bad, meanlp_good, meanlp_bad)`` tuple per pair.
    Both scores come from the *same* forward passes — SLOR and the plain
    length-normalised PLL (mean-LP) differ only by the unigram subtraction, so
    grabbing mean-LP is free. The aggregate caller uses SLOR for ``n_correct``;
    the per-item writer keeps everything.

    Correct means the grammatical sentence wins on SLOR. Ties (exactly equal
    scores) count as wrong — we want strictly better, and exact ties almost
    never happen with float scores anyway.
    """
    from drc.eval.slor import slor_and_meanlp

    n_correct = 0
    scored: list[tuple[str, float, float, float, float]] = []
    for item in items:
        good, ml_good = slor_and_meanlp(
            model, tokenizer, item.good_sentence, unigram_counts,
            mask_batch_size=mask_batch_size, device=device,
        )
        bad, ml_bad = slor_and_meanlp(
            model, tokenizer, item.bad_sentence, unigram_counts,
            mask_batch_size=mask_batch_size, device=device,
        )
        scored.append((item.item_id, good, bad, ml_good, ml_bad))
        if good > bad:
            n_correct += 1
    return len(items), n_correct, scored


def _dose_corpus_path(config: dict[str, Any], config_path: Path,
                      construction: str, dose: str) -> Path:
    """Where the dose corpus for this run lives (matches dose_corpora.py naming)."""
    root = resolve_path(config_path, config["paths"]["dose_corpora"])
    return root / f"{construction}_dose-{dose}.conllu"


def run(
    config_path: Path,
    only_construction: str | None = None,
    mask_batch_size: int = 64,
    per_item: bool = True,
) -> None:
    """Evaluate every discovered model on all four construction test sets.

    With ``per_item`` on (the default) we also stream every pair's two SLOR
    scores to ``results/eval_per_item.csv``. That file feeds the
    memorization-vs-generalization split; the aggregate CSV is untouched either
    way, so existing readers and tests keep working.
    """
    from drc.eval.slor import build_unigram_counts

    config = load_config(config_path)
    models_root = resolve_path(config_path, config["paths"]["models"])
    items_root = resolve_path(config_path, config["paths"]["eval_items"])
    results_root = resolve_path(config_path, config["paths"]["results"])
    tokenizer_path = resolve_path(config_path, config["paths"]["tokenizer"])
    csv_path = results_root / "eval_results.csv"
    per_item_path = results_root / "eval_per_item.csv"

    # Load every construction's minimal pairs once; reused across all models.
    eval_sets: dict[str, list[EvalItem]] = {}
    for construction in CONSTRUCTIONS:
        items_path = items_root / f"{construction}.jsonl"
        if not items_path.exists():
            logger.warning("No eval items for %s at %s; skipping that test set.",
                           construction, items_path)
            continue
        eval_sets[construction] = read_eval_items(items_path)
        logger.info("Loaded %d items for %s.",
                    len(eval_sets[construction]), construction)

    runs = discover_models(models_root)
    if only_construction:
        runs = [r for r in runs if r.construction == only_construction]
        logger.info("Filtered to %d runs for construction=%s.",
                    len(runs), only_construction)

    done = _load_existing_keys(csv_path)
    # Cache unigram counts per (construction, dose) — every seed shares a corpus.
    unigram_cache: dict[tuple[str, str], Any] = {}

    for run_info in runs:
        # Skip the whole model if every test set is already scored for it.
        pending = [
            c for c in eval_sets
            if (run_info.construction, run_info.dose, str(run_info.seed), c)
            not in done
        ]
        if not pending:
            logger.info("Already done: %s dose=%s seed=%d — skipping.",
                        run_info.construction, run_info.dose, run_info.seed)
            continue

        logger.info("Evaluating %s dose=%s seed=%d (%s).",
                    run_info.construction, run_info.dose, run_info.seed,
                    run_info.path.name)
        try:
            model, tokenizer, device = _load_model_and_tokenizer(
                run_info, tokenizer_path
            )
        except Exception as exc:  # noqa: BLE001 - one bad checkpoint shouldn't sink the sweep
            logger.error("Failed to load %s: %s; skipping.",
                         run_info.path.name, exc)
            continue

        key = (run_info.construction, run_info.dose)
        if key not in unigram_cache:
            corpus = _dose_corpus_path(
                config, config_path, run_info.construction, run_info.dose
            )
            if not corpus.exists():
                logger.error(
                    "Dose corpus %s missing; can't build the SLOR unigram "
                    "correction for this model. Skipping.", corpus,
                )
                continue
            unigram_cache[key] = build_unigram_counts(corpus)
        unigram_counts = unigram_cache[key]

        for eval_construction in pending:
            items = eval_sets[eval_construction]
            n_pairs, n_correct, scored = evaluate_pair_set(
                model, tokenizer, device, items, unigram_counts,
                mask_batch_size=mask_batch_size,
            )
            accuracy = n_correct / n_pairs if n_pairs else 0.0
            row = {
                "model_construction": run_info.construction,
                "dose": run_info.dose,
                "seed": run_info.seed,
                "eval_construction": eval_construction,
                "n_pairs": n_pairs,
                "n_correct": n_correct,
                "accuracy": round(accuracy, 6),
                "std_error": round(_binomial_std_error(accuracy, n_pairs), 6),
            }
            # Write the per-item block first, then the aggregate row. The
            # aggregate row is the resume marker: if it's present, the matching
            # per-item block was already flushed, so we never double-write items.
            if per_item:
                _append_per_item_rows(
                    per_item_path,
                    _per_item_rows(run_info, eval_construction, scored),
                )
            _append_row(csv_path, row)
            logger.info("  %s: %d/%d = %.3f",
                        eval_construction, n_correct, n_pairs, accuracy)
            _maybe_sanity_warn(run_info, eval_construction, accuracy)

        # Free the model before the next checkpoint so memory doesn't pile up.
        del model
        _free_memory()

    logger.info("Evaluation complete. Results in %s", csv_path)


def _maybe_sanity_warn(run_info: ModelRun, eval_construction: str,
                       accuracy: float) -> None:
    """Warn if the AANN pilot lands outside its expected replication band."""
    is_pilot = (
        (run_info.construction, run_info.dose, run_info.seed) == SANITY_MODEL
        and eval_construction == SANITY_CONSTRUCTION
    )
    if not is_pilot:
        return
    low, high = SANITY_ACC_RANGE
    if not (low <= accuracy <= high):
        logger.warning(
            "SANITY: AANN pilot (dose=all, seed=42) scored %.3f on AANN, "
            "outside the expected [%.2f, %.2f]. Check the checkpoint and "
            "tokenizer before trusting the sweep.", accuracy, low, high,
        )
    else:
        logger.info("SANITY OK: AANN pilot %.3f within [%.2f, %.2f].",
                    accuracy, low, high)


def _free_memory() -> None:
    """Drop cached CUDA memory between checkpoints. No-op without torch/CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base config with the `paths` section.",
    )
    parser.add_argument(
        "--construction", choices=CONSTRUCTIONS, default=None,
        help="Evaluate only models trained on this construction (default: all).",
    )
    parser.add_argument(
        "--mask-batch-size", type=int, default=64,
        help="Masked positions per forward pass in SLOR. Lower if you hit OOM.",
    )
    parser.add_argument(
        "--per-item", dest="per_item", action="store_true", default=True,
        help="Also write results/eval_per_item.csv with each pair's SLOR scores "
             "(default: on). Feeds the generalization analysis.",
    )
    parser.add_argument(
        "--no-per-item", dest="per_item", action="store_false",
        help="Skip the per-item CSV; write only the aggregate eval_results.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, only_construction=args.construction,
        mask_batch_size=args.mask_batch_size, per_item=args.per_item)


if __name__ == "__main__":
    main()
