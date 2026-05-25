"""Fetch the BabyLM corpora and write them out as plain text.

We pull two things. First, the BabyLM 2024 strict-small set (~10M words) —
that's the corpus every model trains on. Second, a slice of the larger strict
set (~100M words) that we keep around as a replacement pool: when
``dose_corpora`` removes a construction's positive sentences, it backfills with
length- and domain-matched sentences drawn from this pool so the total word
count stays put.

The HuggingFace dataset id is a module constant on purpose. BabyLM's releases
move around between years and mirrors, so when the id goes stale you change it
in one place. If the load fails we raise loudly rather than quietly writing an
empty file — a silently truncated corpus would poison every downstream run.

CLI::

    python -m drc.data.download --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# The strict-small (10M) train split and the larger strict (100M) set we sample
# the replacement pool from. These ids may need updating as BabyLM re-releases;
# keep them here and nowhere else.
BABYLM_10M_DATASET = "cambridge-lti/babylm-2024-strict-small"
BABYLM_100M_DATASET = "cambridge-lti/babylm-2024-strict"

# Target word count for the 10M set and the tolerance band we warn outside of.
TARGET_10M_WORDS = 10_000_000
WORD_COUNT_TOLERANCE = 0.005  # ±0.5%

# How many words to keep in the replacement pool. A few times the 10M set is
# plenty of headroom for the matched-replacement search without dragging the
# whole 100M corpus through parsing later.
REPLACEMENT_POOL_WORDS = 30_000_000


def load_config(config_path: Path) -> dict[str, Any]:
    """Read a YAML config into a dict.

    Thin wrapper so every stage parses configs the same way and we get one
    obvious place to add validation later.
    """
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(config_path: Path, rel: str) -> Path:
    """Turn a config-relative path into an absolute one.

    Paths in ``base.yaml`` are written relative to the repo root (e.g.
    ``data/raw/babylm_10M``). We anchor them to the config file's parent's
    parent — ``configs/base.yaml`` lives one level below the root — so the CLI
    works no matter what directory you launch it from.
    """
    root = config_path.resolve().parent.parent
    candidate = Path(rel)
    return candidate if candidate.is_absolute() else (root / candidate)


# Tab marker that prefixes a source-domain tag on each written line, e.g.
# ``gutenberg\tOnce upon a time ...``. ``parse`` strips this back off and
# carries the domain into the CoNLL-U so dose generation can match on it. We
# keep it on one line because the raw corpus is one record per line.
DOMAIN_SEP = "\t"

# Column names BabyLM has used for the domain/subset label, in preference order.
DOMAIN_COLUMN_CANDIDATES = ("source", "subset", "domain", "split_source")


def _pick_text_column(ds) -> str:
    column = "text"
    if column not in ds.column_names:
        string_cols = [
            c for c in ds.column_names
            if getattr(ds.features[c], "dtype", None) == "string"
        ]
        if not string_cols:
            raise RuntimeError(
                f"Dataset has no text column. Columns present: {ds.column_names}"
            )
        column = string_cols[0]
        logger.warning("No 'text' column; falling back to '%s'.", column)
    return column


def _load_split_records(dataset_id: str, split: str = "train") -> list[tuple[str, str]]:
    """Load a split as ``(source_domain, text)`` pairs.

    ``datasets`` is imported here, not at module top, so the rest of the
    pipeline imports fine in a bare environment. We keep the domain label when
    the release exposes one — dose generation matches replacements on it — and
    fall back to ``"unknown"`` when it's absent rather than guessing.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "The `datasets` package is required to download corpora. "
            "Install it with `pip install datasets`."
        ) from exc

    try:
        ds = load_dataset(dataset_id, split=split)
    except Exception as exc:  # noqa: BLE001 - surface any HF failure clearly
        raise RuntimeError(
            f"Failed to load '{dataset_id}' (split='{split}') from HuggingFace. "
            "Check the dataset id at the top of download.py — BabyLM releases "
            "move between mirrors. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    text_col = _pick_text_column(ds)
    domain_col = next(
        (c for c in DOMAIN_COLUMN_CANDIDATES if c in ds.column_names), None
    )
    if domain_col is None:
        logger.warning(
            "No domain column found among %s; tagging every sentence 'unknown'. "
            "Dose replacement will then match on length only.",
            DOMAIN_COLUMN_CANDIDATES,
        )

    records: list[tuple[str, str]] = []
    domains = ds[domain_col] if domain_col else None
    for i, text in enumerate(ds[text_col]):
        if not text or not text.strip():
            continue
        domain = str(domains[i]) if domains is not None else "unknown"
        records.append((domain, text))
    return records


def _count_words(records: list[tuple[str, str]]) -> int:
    """Whitespace word count over the text field. Matches how BabyLM reports size."""
    return sum(len(text.split()) for _, text in records)


def _write_records(records: list[tuple[str, str]], dest: Path) -> None:
    """Write ``domain<TAB>text`` per record, creating parent dirs as needed.

    A literal newline inside a record would break the one-record-per-line
    contract, so we flatten internal newlines to spaces.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for domain, text in records:
            clean = " ".join(text.split())
            fh.write(f"{domain}{DOMAIN_SEP}{clean}\n")
    logger.info("Wrote %d records to %s", len(records), dest)


def _truncate_to_words(
    records: list[tuple[str, str]], max_words: int
) -> list[tuple[str, str]]:
    """Keep whole records until we hit the word budget.

    Used for the replacement pool: we don't need all 100M words, just a healthy
    slice, and cutting on record boundaries keeps sentences intact.
    """
    out: list[tuple[str, str]] = []
    running = 0
    for domain, text in records:
        n = len(text.split())
        if running + n > max_words:
            break
        out.append((domain, text))
        running += n
    return out


def download_10m(config: dict[str, Any], config_path: Path) -> Path:
    """Download the 10M strict-small set and verify its size."""
    dest = resolve_path(config_path, config["paths"]["raw_corpus"])
    logger.info("Downloading BabyLM-10M from '%s'...", BABYLM_10M_DATASET)
    records = _load_split_records(BABYLM_10M_DATASET)

    n_words = _count_words(records)
    low = TARGET_10M_WORDS * (1 - WORD_COUNT_TOLERANCE)
    high = TARGET_10M_WORDS * (1 + WORD_COUNT_TOLERANCE)
    if not (low <= n_words <= high):
        logger.warning(
            "BabyLM-10M word count is %d, outside the ±%.1f%% band around %d "
            "(%d–%d). The corpus may have changed; double-check the dataset id.",
            n_words, WORD_COUNT_TOLERANCE * 100, TARGET_10M_WORDS,
            int(low), int(high),
        )
    else:
        logger.info("Word count OK: %d (target %d).", n_words, TARGET_10M_WORDS)

    _write_records(records, dest)
    return dest


def download_replacement_pool(
    config: dict[str, Any],
    config_path: Path,
    max_words: int = REPLACEMENT_POOL_WORDS,
) -> Path:
    """Download a slice of the 100M strict set to use as a replacement pool."""
    dest = resolve_path(config_path, config["paths"]["replacement_pool"])
    logger.info(
        "Downloading replacement pool from '%s' (up to %d words)...",
        BABYLM_100M_DATASET, max_words,
    )
    records = _load_split_records(BABYLM_100M_DATASET)
    records = _truncate_to_words(records, max_words)
    logger.info("Replacement pool: %d words across %d records.",
                _count_words(records), len(records))
    _write_records(records, dest)
    return dest


def run(config_path: Path, skip_pool: bool = False) -> None:
    """Run both downloads end to end."""
    config = load_config(config_path)
    download_10m(config, config_path)
    if skip_pool:
        logger.info("Skipping replacement-pool download (--skip-pool).")
    else:
        download_replacement_pool(config, config_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base config with the `paths` section.",
    )
    parser.add_argument(
        "--skip-pool", action="store_true",
        help="Download only the 10M set, not the replacement pool.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, skip_pool=args.skip_pool)


if __name__ == "__main__":
    main()
