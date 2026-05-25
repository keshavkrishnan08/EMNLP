"""Check each construction filter against human (or LLM) judgement.

A filter is only useful if it actually catches what it claims to. So for every
construction we sample 200 sentences the filter flagged and 200 it didn't, drop
them in a CSV, and have someone label whether each one really contains the
construction. From those labels we read off precision and recall. The PRD wants
precision ≥ 0.90 and recall ≥ 0.85 (see ``FilterReport.passes``).

Two ways to get labels:

  * ``--judge manual`` (default) writes the CSV with an empty ``human_label``
    column for you to fill in by hand. Re-run with ``--score`` to read it back.
  * ``--judge llm`` calls Claude to label each sentence. It's a convenience for
    a first pass, not a replacement for human review, and it never invents a
    label — if the SDK or API key is missing, it stops and tells you why.

The audit is a two-phase thing: generate the CSV, label it, then score it.
We never fabricate the human column.

CLI::

    python -m drc.data.qa_audit --config configs/base.yaml --construction aann
    python -m drc.data.qa_audit --config configs/base.yaml --construction aann --score
    python -m drc.data.qa_audit --config configs/base.yaml --construction aann --judge llm
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from pathlib import Path
from typing import Any

from .download import load_config, resolve_path
from .filters.base import FilterReport
from .parse import ParsedSentence, read_with_trees

logger = logging.getLogger(__name__)

SAMPLE_PER_CLASS = 200
AUDIT_SEED = 42

# CSV schema. Keep these names stable — the scorer reads them back by key.
CSV_FIELDS = ("sentence_id", "sentence_text", "filter_label", "human_label", "notes")

# Claude model for the optional LLM judge. Matches the repo-wide model pin.
JUDGE_MODEL = "claude-opus-4-7"


def _get_filter(code: str):
    """Lazy fetch of a construction filter (registry lands in parallel)."""
    from .filters import get_filter
    return get_filter(code)


def _construction_definition(config_path: Path, code: str) -> str:
    """Pull a human-readable definition + example for ``code`` from config.

    Used to brief the LLM judge so it scores against the same definition the
    filter targets, not its own loose intuition about the construction.
    """
    cfg_dir = config_path.resolve().parent
    constructions = load_config(cfg_dir / "constructions.yaml")
    entry = constructions.get(code, {})
    name = entry.get("name", code)
    example = entry.get("example", "")
    return f"{name}. Example: \"{example}\"" if example else name


def sample_sentences(
    parsed_path: Path, code: str, n_per_class: int = SAMPLE_PER_CLASS
) -> list[dict[str, Any]]:
    """Run the filter over the corpus and sample positives and negatives.

    Returns rows ready for the CSV: each carries the sentence, the filter's
    label (1/0), and an empty ``human_label`` for grading. We do reservoir-free
    sampling — collect both classes, then take a seeded random slice of each —
    because we want a fixed, reproducible audit set.
    """
    filt = _get_filter(code)
    positives: list[ParsedSentence] = []
    negatives: list[ParsedSentence] = []

    for sent in read_with_trees(parsed_path):
        if sent.stanza_sentence is None:
            continue
        if filt(sent.stanza_sentence):
            positives.append(sent)
        else:
            negatives.append(sent)

    rng = random.Random(AUDIT_SEED)
    if len(positives) < n_per_class:
        logger.warning("%s: only %d positives, sampling all of them.",
                       code, len(positives))
    if len(negatives) < n_per_class:
        logger.warning("%s: only %d negatives, sampling all of them.",
                       code, len(negatives))

    pos_sample = rng.sample(positives, min(n_per_class, len(positives)))
    neg_sample = rng.sample(negatives, min(n_per_class, len(negatives)))

    rows: list[dict[str, Any]] = []
    for sent in pos_sample:
        rows.append(_row(sent, filter_label=1))
    for sent in neg_sample:
        rows.append(_row(sent, filter_label=0))
    rng.shuffle(rows)  # interleave so a labeler can't pattern-match on order
    return rows


def _row(sent: ParsedSentence, filter_label: int) -> dict[str, Any]:
    return {
        "sentence_id": sent.sent_id,
        "sentence_text": sent.text,
        "filter_label": filter_label,
        "human_label": "",
        "notes": "",
    }


def write_audit_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d audit rows to %s", len(rows), csv_path)


def llm_audit(sentence_text: str, construction_definition: str) -> int:
    """Ask Claude whether a sentence contains the construction. Returns 1/0.

    A convenience judge for a first pass — fast, cheap, and good enough to
    triage obvious cases before a human looks. It is *not* ground truth.

    ``anthropic`` is imported here so this module loads without it. Both a
    missing package and a missing key raise a clear, actionable error; we would
    rather stop than guess a label, since a fabricated judgement would quietly
    corrupt the precision/recall numbers.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "The `anthropic` package is required for --judge llm. Install it "
            "with `pip install anthropic`, or use --judge manual."
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your key or use --judge manual."
        )

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are auditing a grammatical-construction detector. Decide whether "
        "the sentence below contains the target construction.\n\n"
        f"Target construction: {construction_definition}\n\n"
        f"Sentence: {sentence_text}\n\n"
        "Answer with exactly one character: 1 if the sentence contains the "
        "construction, 0 if it does not. No explanation."
    )
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    ).strip()
    if text.startswith("1"):
        return 1
    if text.startswith("0"):
        return 0
    raise RuntimeError(
        f"LLM judge returned an unparseable answer: {text!r}. Refusing to guess."
    )


def fill_llm_labels(rows: list[dict[str, Any]], definition: str) -> None:
    """Populate ``human_label`` in place using the LLM judge."""
    for i, row in enumerate(rows, start=1):
        row["human_label"] = llm_audit(row["sentence_text"], definition)
        if i % 50 == 0:
            logger.info("LLM judged %d/%d rows...", i, len(rows))


def score_audit(csv_path: Path, code: str) -> FilterReport:
    """Read a labeled CSV and compute precision/recall into a FilterReport.

    Treats the filter as the system under test and ``human_label`` as truth.
    Skips rows whose human label is still blank, and tells you how many it
    skipped so a half-labeled file can't masquerade as a finished audit.
    """
    tp = fp = fn = tn = 0
    skipped = 0
    notes: list[str] = []

    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            human = str(row.get("human_label", "")).strip()
            if human not in {"0", "1"}:
                skipped += 1
                continue
            pred = int(str(row["filter_label"]).strip())
            truth = int(human)
            if pred == 1 and truth == 1:
                tp += 1
            elif pred == 1 and truth == 0:
                fp += 1
            elif pred == 0 and truth == 1:
                fn += 1
            else:
                tn += 1

    n_audited = tp + fp + fn + tn
    if skipped:
        notes.append(f"{skipped} rows skipped (human_label not yet filled).")
    if n_audited == 0:
        raise RuntimeError(
            f"No labeled rows in {csv_path}. Fill the human_label column "
            "(0/1) before scoring."
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    report = FilterReport(
        code=code,
        n_positive=tp + fp,
        precision=precision,
        recall=recall,
        n_audited=n_audited,
        notes=notes,
    )
    logger.info(
        "%s audit: precision=%.3f recall=%.3f (n=%d, %s)",
        code, precision, recall, n_audited,
        "PASS" if report.passes else "FAIL",
    )
    return report


def run(
    config_path: Path,
    code: str,
    judge: str = "manual",
    score: bool = False,
) -> None:
    """Generate (and optionally LLM-label or score) one construction's audit."""
    config: dict[str, Any] = load_config(config_path)
    parsed_path = resolve_path(config_path, config["paths"]["parsed"])
    # The .gitignore reserves data/qa_audits/ for these; honour that layout.
    audit_dir = resolve_path(config_path, "data/qa_audits")
    csv_path = audit_dir / f"{code}_audit.csv"

    if score:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No audit CSV at {csv_path}. Generate it first (drop --score)."
            )
        score_audit(csv_path, code)
        return

    if not parsed_path.exists():
        raise FileNotFoundError(
            f"Parsed corpus not found at {parsed_path}. Run `python -m drc.data.parse` "
            "first."
        )

    rows = sample_sentences(parsed_path, code)

    if judge == "llm":
        definition = _construction_definition(config_path, code)
        logger.info("LLM-judging %d rows for %s with %s...",
                    len(rows), code, JUDGE_MODEL)
        fill_llm_labels(rows, definition)

    write_audit_csv(rows, csv_path)
    if judge == "manual":
        logger.info(
            "Fill the human_label column (0/1) in %s, then re-run with --score.",
            csv_path,
        )
    else:
        # LLM labels are in; we can score straight away, but flag they're not human.
        logger.info("LLM labels written. Scoring now (LLM judge, not ground truth).")
        score_audit(csv_path, code)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base config with the `paths` section.",
    )
    parser.add_argument(
        "--construction", required=True,
        help="Construction code to audit (e.g. aann).",
    )
    parser.add_argument(
        "--judge", choices=("manual", "llm"), default="manual",
        help="How to label: hand-label the CSV (manual) or call Claude (llm).",
    )
    parser.add_argument(
        "--score", action="store_true",
        help="Score an already-labeled CSV instead of generating a new one.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, code=args.construction, judge=args.judge, score=args.score)


if __name__ == "__main__":
    main()
