"""Train the one Byte-Level BPE tokenizer the whole sweep shares.

Every model in the experiment reads the same vocabulary. If each run trained its
own tokenizer the perplexities wouldn't be comparable across doses — different
segmentation, different loss surface — so we fit a single tokenizer once, freeze
it, and point all 60 runs at it.

We train on the canonical corpus: the AANN dose=all corpus built with corpus
seed 42. That's the largest, least-surgically-altered corpus in the set, which
makes it the natural reference for vocabulary. The input file is a CLI arg, so
you can repoint it if the canonical choice ever changes, but the default is
wired to that file.

The corpus on disk is CoNLL-U (one parsed sentence per block, with a
``# text =`` header). We pull the raw text back out of those headers and feed it
to the BPE trainer line by line, so the tokenizer learns from natural sentences
rather than tokenised columns.

CLI::

    python -m drc.tokenizer.train_tokenizer --config configs/base.yaml
    python -m drc.tokenizer.train_tokenizer --corpus data/dose_corpora/aann_dose-all.conllu
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from drc.data.download import load_config, resolve_path

logger = logging.getLogger(__name__)

# Frozen tokenizer spec. These mirror the BabyLM/LTG-BERT setup and must not
# drift: the vocab size in particular has to match model.vocab_size in base.yaml,
# or the embedding matrix won't line up with the token ids.
VOCAB_SIZE = 16384
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<mask>", "<unk>"]

# The construction/dose/corpus-seed that defines the canonical corpus. Used only
# to build the default input path from the config's dose_corpora directory.
CANONICAL_CONSTRUCTION = "aann"
CANONICAL_DOSE = "all"

# Acceptance band for the average tokens-per-word sanity check. A healthy
# subword vocabulary over English sits a little above one token per word; way
# above 1.4 means the vocab is too small or the corpus is off.
TOKENS_PER_WORD_RANGE = (1.2, 1.4)

# A few held-out sentences we round-trip to prove encode/decode is lossless.
_ROUND_TRIP_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore on a sunny afternoon.",
    "An astonishingly beautiful three days followed the storm.",
    "Whether the model learns the construction is exactly the question.",
    "Colorless green ideas sleep furiously, or so the saying goes.",
]


def default_corpus_path(config: dict[str, Any], config_path: Path) -> Path:
    """Build the path to the canonical training corpus from the config.

    Matches the filename convention in :mod:`drc.data.dose_corpora`
    (``{construction}_dose-{dose}.conllu``).
    """
    dose_root = resolve_path(config_path, config["paths"]["dose_corpora"])
    return dose_root / f"{CANONICAL_CONSTRUCTION}_dose-{CANONICAL_DOSE}.conllu"


def iter_corpus_text(corpus_path: Path) -> Iterator[str]:
    """Yield one natural-language sentence per CoNLL-U block.

    We read the ``# text =`` header rather than re-joining the FORM column, so
    the tokenizer sees the original punctuation and spacing. Reuses the parse
    module's reader so there's a single source of truth for the CoNLL-U format.
    """
    from drc.data.parse import read_metadata

    for sent in read_metadata(corpus_path):
        text = sent.text.strip()
        if text:
            yield text


def build_tokenizer():
    """Construct an untrained Byte-Level BPE tokenizer.

    Kept separate from training so tests can build one cheaply. ``tokenizers``
    is imported here so the module stays importable without it installed.
    """
    from tokenizers import ByteLevelBPETokenizer

    return ByteLevelBPETokenizer()


def train(corpus_path: Path, out_dir: Path) -> Path:
    """Train the BPE tokenizer on ``corpus_path`` and save it to ``out_dir``.

    Saves the legacy pair (``vocab.json`` + ``merges.txt``) and the unified
    ``tokenizer.json``. The transformers ``PreTrainedTokenizerFast`` loader the
    training code uses prefers the single-file form, but we write both so the
    artifacts are usable either way.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Training corpus not found at {corpus_path}. Build the dose corpora "
            "first with `python -m drc.data.dose_corpora`."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()

    logger.info("Training Byte-Level BPE (vocab=%d) on %s...", VOCAB_SIZE, corpus_path)
    tokenizer.train_from_iterator(
        iter_corpus_text(corpus_path),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    # Legacy artifacts (vocab.json, merges.txt) for tooling that wants them...
    tokenizer.save_model(str(out_dir))
    # ...and the single-file form the loader actually consumes.
    tokenizer_json = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    logger.info("Saved tokenizer to %s.", out_dir)
    return tokenizer_json


def _load_fast_tokenizer(out_dir: Path):
    """Reload the saved tokenizer through transformers for the sanity checks."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_file=str(out_dir / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        unk_token="<unk>",
    )


def sanity_check(out_dir: Path, corpus_path: Path, sample_sentences: int = 2000) -> dict:
    """Verify the saved tokenizer: vocab size, round-trips, tokens-per-word.

    Returns a small report dict and raises ``AssertionError`` on any failure so
    a broken tokenizer can't slip silently into 60 downstream runs. The
    tokens-per-word figure is measured on a sample of corpus sentences, not the
    five canned ones, so it reflects real text.
    """
    tok = _load_fast_tokenizer(out_dir)

    vocab_size = tok.vocab_size
    assert vocab_size == VOCAB_SIZE, (
        f"Vocab size is {vocab_size}, expected {VOCAB_SIZE}."
    )

    # Round-trip: encode then decode should recover the sentence's content. BPE
    # decode can shift whitespace, so compare on collapsed whitespace.
    for sent in _ROUND_TRIP_SENTENCES:
        ids = tok.encode(sent)
        decoded = tok.decode(ids, skip_special_tokens=True)
        if " ".join(decoded.split()) != " ".join(sent.split()):
            raise AssertionError(
                f"Round-trip failed.\n  in:  {sent!r}\n  out: {decoded!r}"
            )

    # Average tokens per word over a corpus sample.
    total_tokens = 0
    total_words = 0
    for i, text in enumerate(iter_corpus_text(corpus_path)):
        if i >= sample_sentences:
            break
        n_words = len(text.split())
        if n_words == 0:
            continue
        total_tokens += len(tok.encode(text))
        total_words += n_words

    tokens_per_word = (total_tokens / total_words) if total_words else float("nan")
    lo, hi = TOKENS_PER_WORD_RANGE
    assert lo <= tokens_per_word <= hi, (
        f"Avg tokens/word is {tokens_per_word:.3f}, outside [{lo}, {hi}]. "
        "The vocabulary or corpus is probably wrong."
    )

    report = {
        "vocab_size": vocab_size,
        "round_trip_sentences": len(_ROUND_TRIP_SENTENCES),
        "tokens_per_word": round(tokens_per_word, 4),
        "sample_sentences": min(sample_sentences, total_words and i + 1 or 0),
    }
    logger.info(
        "Sanity OK: vocab=%d, %d round-trips, tokens/word=%.3f.",
        vocab_size, len(_ROUND_TRIP_SENTENCES), tokens_per_word,
    )
    return report


def run(
    config_path: Path,
    corpus_path: Path | None = None,
    skip_sanity: bool = False,
) -> Path:
    """Train and (unless skipped) sanity-check the shared tokenizer."""
    config = load_config(config_path)
    out_dir = resolve_path(config_path, config["paths"]["tokenizer"])
    corpus = corpus_path or default_corpus_path(config, config_path)

    train(corpus, out_dir)
    if not skip_sanity:
        sanity_check(out_dir, corpus)
    return out_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base config with the `paths` section.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=None,
        help="CoNLL-U corpus to train on (default: the canonical AANN dose=all).",
    )
    parser.add_argument(
        "--skip-sanity", action="store_true",
        help="Skip the post-training sanity checks (not recommended).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, corpus_path=args.corpus, skip_sanity=args.skip_sanity)


if __name__ == "__main__":
    main()
