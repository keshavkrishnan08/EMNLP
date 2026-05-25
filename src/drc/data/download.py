"""Fetch the BabyLM corpora and write them out as plain text.

We pull two things. First, the BabyLM strict-small set (~10M words) — that's the
corpus every model trains on. Second, a slice of the larger strict set (~100M
words) that we keep as a replacement pool: when ``dose_corpora`` removes a
construction's positive sentences, it backfills with length- and domain-matched
sentences from this pool so the total word count stays put.

How the data is shaped (verified against the official release): BabyLM ships as
**one plain-text file per source domain**, named ``<domain>.train.txt`` (e.g.
``childes.train.txt``, ``open_subtitles.train.txt``). The filename *is* the
domain label, which is exactly what dose replacement needs. So we don't go
through ``datasets.load_dataset`` and its custom loading script — that path is
brittle on modern ``datasets`` versions. Instead we grab the raw files straight
from the Hub with ``huggingface_hub`` and tag each line with the domain taken
from its filename.

The repo ids are constants (overridable in the config under a ``data:`` block).
BabyLM re-releases every year and the org/repo names shift, so when an id goes
stale you change it in one place. The official data is linked from
https://babylm.github.io — if a repo 404s, check there for the current edition.
If a download fails we raise loudly rather than quietly writing an empty file; a
silently truncated corpus would poison every downstream run.

CLI::

    python -m drc.data.download --config configs/base.yaml
    python -m drc.data.download --config configs/base.yaml --strict-small-repo OTHER/REPO
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Official BabyLM data repos on the Hugging Face Hub (dataset repos). The
# strict-small repo is the 10M training set; the strict repo is the 100M set we
# slice the replacement pool from. Override per-run via the config's `data:`
# block or the CLI flags rather than editing here.
BABYLM_STRICT_SMALL_REPO = "BabyLM-community/BabyLM-2026-Strict-Small"
BABYLM_STRICT_REPO = "BabyLM-community/BabyLM-2026-Strict"

# We only want the training text files. Both extensions appear across editions
# (2024 used ``.train``; 2026 uses ``.train.txt``), so we accept either.
TRAIN_FILE_PATTERNS = ("*.train.txt", "*.train")

# Target word count for the 10M set and the band we warn outside of. Editions
# vary by a few percent, so this is a sanity check, not a hard gate.
TARGET_10M_WORDS = 10_000_000
WORD_COUNT_TOLERANCE = 0.05  # ±5%

# How many words to keep in the replacement pool. A few times the 10M set is
# plenty of headroom for the matched-replacement search without dragging the
# whole 100M corpus through parsing later.
REPLACEMENT_POOL_WORDS = 30_000_000

# Tab marker prefixing a source-domain tag on each written line, e.g.
# ``gutenberg\tOnce upon a time ...``. ``parse`` strips this back off and carries
# the domain into the CoNLL-U so dose generation can match on it.
DOMAIN_SEP = "\t"


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
    ``data/raw/babylm_10M.txt``). We anchor them to the config file's parent's
    parent — ``configs/base.yaml`` lives one level below the root — so the CLI
    works no matter which directory you launch it from.
    """
    root = config_path.resolve().parent.parent
    candidate = Path(rel)
    return candidate if candidate.is_absolute() else (root / candidate)


def _repo_for(config: dict[str, Any], key: str, default: str) -> str:
    """Look up a repo id in the config's optional ``data:`` block, else default."""
    return str(config.get("data", {}).get(key, default))


def _domain_from_filename(filename: str) -> str:
    """``open_subtitles.train.txt`` -> ``open_subtitles``.

    Strip the ``.train`` / ``.train.txt`` suffix; whatever's left is the source
    domain. We keep it lowercase and underscored, matching the file naming.
    """
    name = Path(filename).name
    for suffix in (".train.txt", ".train"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _download_train_files(repo_id: str) -> list[Path]:
    """Download the per-domain training files from a dataset repo.

    Uses ``huggingface_hub`` directly (no ``datasets`` loading script). Imported
    lazily so the rest of the pipeline imports fine without it. Returns the local
    paths of the matched ``*.train(.txt)`` files.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "The `huggingface_hub` package is required to download corpora. "
            "Install it with `pip install huggingface_hub` (it also ships with "
            "`datasets` and `transformers`)."
        ) from exc

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=list(TRAIN_FILE_PATTERNS),
        )
    except Exception as exc:  # noqa: BLE001 - surface any Hub failure clearly
        raise RuntimeError(
            f"Failed to download '{repo_id}' from the Hugging Face Hub. "
            "Check the repo id (see the constants in download.py or the `data:` "
            "block in your config) — the current edition is linked from "
            f"https://babylm.github.io. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    files: list[Path] = []
    for pattern in TRAIN_FILE_PATTERNS:
        files.extend(sorted(Path(local_dir).rglob(pattern)))
    # rglob across both patterns can double-count a ``.train.txt`` (it also ends
    # with ``.txt`` but not ``.train``); de-dupe while preserving order.
    seen: set[Path] = set()
    unique = [f for f in files if not (f in seen or seen.add(f))]
    if not unique:
        raise RuntimeError(
            f"Downloaded '{repo_id}' but found no files matching {TRAIN_FILE_PATTERNS}. "
            "The release layout may have changed; inspect the repo file tree."
        )
    logger.info("Found %d training files in '%s': %s",
                len(unique), repo_id, ", ".join(f.name for f in unique))
    return unique


def _records_from_files(files: list[Path]) -> list[tuple[str, str]]:
    """Read ``(domain, text)`` records, one per non-empty line, across all files."""
    records: list[tuple[str, str]] = []
    for path in files:
        domain = _domain_from_filename(path.name)
        n_before = len(records)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if text:
                    records.append((domain, text))
        logger.info("  %-20s %8d lines", domain, len(records) - n_before)
    return records


def _count_words(records: list[tuple[str, str]]) -> int:
    """Whitespace word count over the text field. Matches how BabyLM reports size."""
    return sum(len(text.split()) for _, text in records)


def _write_records(records: list[tuple[str, str]], dest: Path) -> None:
    """Write ``domain<TAB>text`` per record, creating parent dirs as needed.

    A literal newline inside a record would break the one-record-per-line
    contract, so we flatten internal whitespace to single spaces.
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
    repo = _repo_for(config, "strict_small_repo", BABYLM_STRICT_SMALL_REPO)
    dest = resolve_path(config_path, config["paths"]["raw_corpus"])
    logger.info("Downloading BabyLM strict-small (10M) from '%s'...", repo)
    records = _records_from_files(_download_train_files(repo))

    n_words = _count_words(records)
    low = TARGET_10M_WORDS * (1 - WORD_COUNT_TOLERANCE)
    high = TARGET_10M_WORDS * (1 + WORD_COUNT_TOLERANCE)
    if not (low <= n_words <= high):
        logger.warning(
            "Strict-small word count is %d, outside the ±%.0f%% band around %d "
            "(%d-%d). The release may have changed; double-check the repo id.",
            n_words, WORD_COUNT_TOLERANCE * 100, TARGET_10M_WORDS, int(low), int(high),
        )
    else:
        logger.info("Word count OK: %d (target ~%d).", n_words, TARGET_10M_WORDS)

    _write_records(records, dest)
    return dest


def download_replacement_pool(
    config: dict[str, Any],
    config_path: Path,
    max_words: int = REPLACEMENT_POOL_WORDS,
) -> Path:
    """Download a slice of the 100M strict set to use as a replacement pool."""
    repo = _repo_for(config, "strict_repo", BABYLM_STRICT_REPO)
    dest = resolve_path(config_path, config["paths"]["replacement_pool"])
    logger.info("Downloading replacement pool from '%s' (up to %d words)...", repo, max_words)
    records = _records_from_files(_download_train_files(repo))
    records = _truncate_to_words(records, max_words)
    logger.info("Replacement pool: %d words across %d records.",
                _count_words(records), len(records))
    _write_records(records, dest)
    return dest


def run(
    config_path: Path,
    skip_pool: bool = False,
    strict_small_repo: str | None = None,
    strict_repo: str | None = None,
) -> None:
    """Run both downloads end to end, with optional repo-id overrides."""
    config = load_config(config_path)
    # CLI overrides win over the config's `data:` block, which wins over the
    # module defaults.
    data_cfg = dict(config.get("data", {}))
    if strict_small_repo:
        data_cfg["strict_small_repo"] = strict_small_repo
    if strict_repo:
        data_cfg["strict_repo"] = strict_repo
    config["data"] = data_cfg

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
    parser.add_argument(
        "--strict-small-repo", default=None,
        help="Override the 10M dataset repo id (else config `data:` / default).",
    )
    parser.add_argument(
        "--strict-repo", default=None,
        help="Override the 100M dataset repo id used for the replacement pool.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(
        args.config,
        skip_pool=args.skip_pool,
        strict_small_repo=args.strict_small_repo,
        strict_repo=args.strict_repo,
    )


if __name__ == "__main__":
    main()
