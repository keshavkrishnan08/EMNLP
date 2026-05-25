"""Build the dose-level corpora — the heart of the experiment.

For each construction we want a family of corpora that differ in exactly one
thing: how many times the target construction shows up. Dose 0 sees it zero
times, dose 4 sees it four, then 16, 64, and finally "all" (every attested
instance left untouched). Hold everything else constant and the model's
behaviour across doses traces out a dose-response curve.

The tricky part is keeping "everything else" constant. If we just deleted the
extra positive sentences, the corpus would shrink and its genre mix would
drift, and then we couldn't tell whether a change in behaviour came from the
construction or from the smaller, lopsided training set. So we replace every
removed sentence with a matched one from the 100M pool:

  1. same source domain (a news sentence for a news sentence),
  2. length within ±20%, relaxing to ±40% if nothing fits,
  3. and if even that's empty, the nearest-length sentence from the same domain.

The whole thing runs off one fixed corpus seed (42). The dose corpora are
generated once and frozen; training seeds vary the model, not the data. Every
corpus gets an audit JSON recording exactly which sentences were kept, removed,
and swapped in, and we re-run the construction filter afterwards to prove the
counts came out right.

CLI::

    python -m drc.data.dose_corpora --config configs/base.yaml
    python -m drc.data.dose_corpora --config configs/base.yaml --construction aann
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from .download import load_config, resolve_path
from .parse import ParsedSentence, read_metadata, read_with_trees

logger = logging.getLogger(__name__)

# One fixed seed for all corpus construction. Training randomness lives
# elsewhere (drc.SEEDS); the data itself must be byte-for-byte reproducible.
CORPUS_SEED = 42

# Length-match bands, tried in order before we fall back to nearest-neighbour.
LENGTH_BANDS = (0.20, 0.40)

# Allowed drift in total word count relative to the source corpus.
WORD_COUNT_TOLERANCE = 0.005  # ±0.5%


def _dose_levels() -> tuple:
    from drc import DOSE_LEVELS
    return DOSE_LEVELS


def _constructions() -> tuple:
    from drc import CONSTRUCTIONS
    return CONSTRUCTIONS


def _get_filter(code: str):
    """Fetch a construction filter by code, lazily.

    The filters package is being built in parallel, so we import it inside the
    function. That keeps this module importable even before the registry lands.
    """
    from .filters import get_filter
    return get_filter(code)


def find_positives(
    sentences: list[ParsedSentence], code: str
) -> list[int]:
    """Return the sent_ids of every sentence the filter flags for ``code``.

    Needs parse trees, so callers pass sentences read via ``read_with_trees``.
    """
    filt = _get_filter(code)
    hits: list[int] = []
    for sent in sentences:
        if sent.stanza_sentence is None:
            continue
        if filt(sent.stanza_sentence):
            hits.append(sent.sent_id)
    return hits


def _stable_index(key: int, n: int) -> int:
    """A deterministic index in ``[0, n)`` from an integer key.

    We hash explicitly with md5 rather than Python's built-in ``hash``, whose
    salt varies between interpreter runs — reproducibility across machines is
    the whole point of the corpus seed.
    """
    digest = hashlib.md5(f"{CORPUS_SEED}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % max(1, n)


def _choose_replacement(
    removed: ParsedSentence,
    pool_by_domain: dict[str, list[ParsedSentence]],
) -> ParsedSentence | None:
    """Pick a pool sentence to stand in for a removed positive.

    The choice is a deterministic function of the removed slot *alone* — same
    domain, the tightest length band that has candidates, tie-broken by length
    and sent_id, then indexed by a stable hash of the removed sentence's id. It
    does **not** depend on the dose or on which other slots were removed.

    That independence is what makes the dose ladder strictly nested: a slot
    removed at both dose=4 and dose=16 gets the *same* filler, so the only thing
    that differs between two doses is the handful of slots that flip from filler
    back to a real instance. We deliberately allow a pool sentence to be reused
    across slots (negligible at 10M words) rather than let a global "used" set
    leak dose information into the corpus and reintroduce the confound.
    """
    candidates = pool_by_domain.get(removed.source_domain, [])
    if not candidates:
        return None

    target = removed.word_count
    chosen = candidates
    for band in LENGTH_BANDS:
        lo, hi = target * (1 - band), target * (1 + band)
        in_band = [s for s in candidates if lo <= s.word_count <= hi]
        if in_band:
            chosen = in_band
            break

    ordered = sorted(chosen, key=lambda s: (abs(s.word_count - target), s.sent_id))
    return ordered[_stable_index(removed.sent_id, len(ordered))]


def _index_pool(pool: list[ParsedSentence]) -> dict[str, list[ParsedSentence]]:
    by_domain: dict[str, list[ParsedSentence]] = {}
    for sent in pool:
        by_domain.setdefault(sent.source_domain, []).append(sent)
    return by_domain


def _write_full_corpus(base_sentences: list[ParsedSentence], out_path: Path) -> None:
    """Write the unfiltered corpus verbatim --- the shared E_max reference.

    No construction is removed, so this is just the original parsed corpus in
    CoNLL-U. One of these (per seed) is the ceiling point for every construction.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for sent in base_sentences:
            block = sent.block
            fh.write(block if block.endswith("\n") else block + "\n")
            fh.write("\n")


def build_corpus(
    code: str,
    dose,
    base_sentences: list[ParsedSentence],
    positive_ids: list[int],
    pool: list[ParsedSentence],
    out_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Generate one (construction, dose) corpus and its audit record.

    ``base_sentences`` is the full parsed corpus in order; ``positive_ids`` are
    the construction's hits within it. We keep ``dose`` of those positives,
    remove the rest, and swap a matched pool sentence in for each removal so the
    word count barely moves. Writes the corpus as CoNLL-U and returns the audit.
    """
    rng = random.Random(CORPUS_SEED)
    pool_by_domain = _index_pool(pool)

    # Decide which positives to keep. Sort first so the seeded shuffle is
    # deterministic regardless of the order sentences came off disk.
    ordered_pos = sorted(positive_ids)
    if dose == "all":
        keep_ids = set(ordered_pos)
        remove_ids: list[int] = []
    else:
        shuffled = ordered_pos[:]
        rng.shuffle(shuffled)
        n_keep = min(int(dose), len(shuffled))
        if n_keep < int(dose):
            logger.warning(
                "%s dose=%s: only %d positives attested, keeping all of them.",
                code, dose, len(shuffled),
            )
        keep_ids = set(shuffled[:n_keep])
        remove_ids = shuffled[n_keep:]

    removed_set = set(remove_ids)
    by_id = {s.sent_id: s for s in base_sentences}

    replacement_assignments: dict[str, int] = {}  # removed_id -> pool sent_id
    unmatched: list[int] = []

    # Stream the corpus in original order, substituting as we hit a removal.
    # Each filler is a deterministic function of its own slot (not the dose), so
    # the ladder is strictly nested: corpus(dose=k) and corpus(dose=k') differ
    # only in the slots that flip between a filler and a real instance.
    out_blocks: list[str] = []
    for sent in base_sentences:
        if sent.sent_id in removed_set:
            repl = _choose_replacement(sent, pool_by_domain)
            if repl is None:
                # No domain-matched replacement available. Drop the sentence and
                # record it honestly rather than padding the count with junk.
                unmatched.append(sent.sent_id)
                continue
            replacement_assignments[str(sent.sent_id)] = repl.sent_id
            out_blocks.append(repl.block)
        else:
            out_blocks.append(sent.block)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for block in out_blocks:
            fh.write(block if block.endswith("\n") else block + "\n")
            fh.write("\n")

    kept_words = sum(by_id[i].word_count for i in keep_ids)
    audit = {
        "construction": code,
        "dose": dose,
        "corpus_seed": CORPUS_SEED,
        "n_positive_total": len(positive_ids),
        "n_kept": len(keep_ids),
        "n_removed": len(remove_ids),
        "n_replaced": len(replacement_assignments),
        "n_unmatched": len(unmatched),
        "kept_positive_ids": sorted(keep_ids),
        "removed_positive_ids": sorted(removed_set),
        "replacement_assignments": replacement_assignments,
        "unmatched_removed_ids": unmatched,
        "kept_positive_words": kept_words,
        "output_corpus": str(out_path),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    logger.info(
        "%s dose=%s -> %s (kept %d, removed %d, replaced %d, unmatched %d)",
        code, dose, out_path.name, len(keep_ids), len(remove_ids),
        len(replacement_assignments), len(unmatched),
    )
    return audit


def _expected_count(dose, n_total: int) -> int:
    """How many positives a finished dose corpus should contain."""
    if dose == "all":
        return n_total
    return min(int(dose), n_total)


def sanity_check_corpus(corpus_path: Path, code: str) -> int:
    """Re-run the filter on a generated corpus and return the positive count.

    This is the proof that the surgery worked: a dose-4 corpus had better
    contain exactly four hits. We rebuild trees from the corpus's own CoNLL-U so
    the check is fully independent of how the corpus was assembled.
    """
    filt = _get_filter(code)
    count = 0
    for sent in read_with_trees(corpus_path):
        if sent.stanza_sentence is None:
            continue
        if filt(sent.stanza_sentence):
            count += 1
    return count


def run(
    config_path: Path,
    only_construction: str | None = None,
    skip_sanity: bool = False,
) -> None:
    """Generate every dose corpus (or just one construction's) and verify them."""
    config: dict[str, Any] = load_config(config_path)
    parsed_path = resolve_path(config_path, config["paths"]["parsed"])
    pool_parsed = resolve_path(config_path, config["paths"]["parsed_pool"])
    out_root = resolve_path(config_path, config["paths"]["dose_corpora"])
    results_root = resolve_path(config_path, config["paths"]["results"])

    if not parsed_path.exists():
        raise FileNotFoundError(
            f"Parsed corpus not found at {parsed_path}. Run `python -m drc.data.parse` "
            "first."
        )
    if not pool_parsed.exists():
        raise FileNotFoundError(
            f"Parsed replacement pool not found at {pool_parsed}. Run "
            "`python -m drc.data.parse` (it parses both the corpus and the pool) "
            "after downloading the pool without --skip-pool."
        )

    from collections import defaultdict

    from drc.design import FULL_CORPUS_CODE, dose_cells

    # The tiered design decides which (construction, dose) corpora to build; we
    # group by construction so positives are located once each.
    cells = dose_cells(config)
    if only_construction:
        cells = [(c, d) for (c, d) in cells if c == only_construction]
    doses_by_construction: dict[str, list] = defaultdict(list)
    for c, d in cells:
        doses_by_construction[c].append(d)

    logger.info("Loading parsed corpus with trees from %s...", parsed_path)
    base_sentences = list(read_with_trees(parsed_path))
    logger.info("Loading replacement pool (metadata only) from %s...", pool_parsed)
    pool = list(read_metadata(pool_parsed))

    sanity: dict[str, Any] = {}
    for code, doses in doses_by_construction.items():
        # The shared full-corpus model trains on the unfiltered corpus: write
        # every sentence through untouched, no construction removed.
        if code == FULL_CORPUS_CODE:
            out_path = out_root / f"{code}_dose-all.conllu"
            _write_full_corpus(base_sentences, out_path)
            logger.info("Wrote shared full corpus -> %s (%d sentences).",
                        out_path.name, len(base_sentences))
            continue

        logger.info("Finding %s positives in base corpus...", code)
        positive_ids = find_positives(base_sentences, code)
        logger.info("%s: %d positives attested.", code, len(positive_ids))

        for dose in doses:
            out_path = out_root / f"{code}_dose-{dose}.conllu"
            audit_path = out_root / f"{code}_dose-{dose}.audit.json"
            build_corpus(
                code, dose, base_sentences, positive_ids, pool,
                out_path, audit_path,
            )

            if skip_sanity:
                continue
            observed = sanity_check_corpus(out_path, code)
            expected = _expected_count(dose, len(positive_ids))
            ok = observed == expected
            sanity[f"{code}_dose-{dose}"] = {
                "expected": expected, "observed": observed, "pass": ok,
            }
            if ok:
                logger.info("Sanity OK: %s dose=%s -> %d positives.",
                            code, dose, observed)
            else:
                logger.error(
                    "Sanity FAIL: %s dose=%s expected %d, got %d.",
                    code, dose, expected, observed,
                )

    if not skip_sanity:
        sanity_path = results_root / "dose_corpora_sanity.json"
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sanity_path, "w", encoding="utf-8") as fh:
            json.dump(sanity, fh, indent=2)
        n_fail = sum(1 for v in sanity.values() if not v["pass"])
        if n_fail:
            logger.error("%d dose corpora failed the count sanity check.", n_fail)
        logger.info("Sanity results written to %s.", sanity_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base config with the `paths` section.",
    )
    parser.add_argument(
        "--construction", default=None,
        help="Only build corpora for this construction code (default: all four).",
    )
    parser.add_argument(
        "--skip-sanity", action="store_true",
        help="Skip the post-hoc filter re-run (faster, but unverified).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, only_construction=args.construction, skip_sanity=args.skip_sanity)


if __name__ == "__main__":
    main()
