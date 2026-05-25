"""The deflationary baseline: does the dose-response survive with only n-grams?

The neural results are the headline, but a skeptic will ask whether a model is
really *learning the construction* or just tracking local word co-occurrence.
So we run the same minimal-pair test with a plain 4-gram language model trained
on each dose corpus. If accuracy still climbs with dose under n-grams alone, the
"learning" story is partly deflated — the signal was recoverable from surface
statistics. If it doesn't, that's evidence the neural model is doing something
n-grams can't.

Smoothing: interpolated add-k (Jelinek-Mercer-style backoff). For each order we
mix the maximum-likelihood estimate with the next lower order, and the unigram
sits on add-k over the vocabulary so nothing is ever zero. We pick simple,
documented smoothing over a finicky Kneser-Ney implementation on purpose — the
baseline only has to be honest and reproducible, not state of the art. The
interpolation weights and k are module constants below.

If KenLM is installed we *could* use it, but it's an optional heavy dep; the
pure-Python path is the default and needs nothing beyond the standard library.

Same accuracy rule as the neural eval: a pair is correct when the good sentence
gets the higher length-normalised log-probability. Same CSV schema, written to
``results/ngram_baseline.csv``.

CLI::

    python -m drc.eval.ngram_baseline --config configs/base.yaml
    python -m drc.eval.ngram_baseline --config configs/base.yaml --construction aann
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from drc import CONSTRUCTIONS, DOSE_LEVELS, SEEDS
from drc.data.download import load_config, resolve_path

logger = logging.getLogger(__name__)

# n-gram order. Four matches the "4-gram baseline" in the paper.
NGRAM_ORDER = 4

# add-k mass for the unigram floor. Small, so seen words dominate, but nonzero
# so unseen words get a finite probability.
ADD_K = 0.01

# Interpolation weights from the top order down. lambda_4 * p4 + lambda_3 * p3 +
# ... They sum to 1. Heavier weight up top, with enough lower-order mass to back
# off gracefully when the high-order context was never seen.
INTERP_WEIGHTS = (0.5, 0.25, 0.15, 0.10)

# Sentence boundary tokens. We pad each sentence with (order-1) <s> on the left
# and a single </s> on the right so the model scores where sentences start/end.
BOS = "<s>"
EOS = "</s>"

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Same schema as run_eval so analysis code can treat both files identically.
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


def _tokenize(sentence: str) -> list[str]:
    """Lowercase word tokens. Matches the SLOR unigram tokeniser for consistency."""
    return _WORD_RE.findall(sentence.lower())


class EvalItem(NamedTuple):
    """One minimal pair (mirrors run_eval.EvalItem; kept local to stay standalone)."""

    item_id: str
    construction: str
    good_sentence: str
    bad_sentence: str


class NgramLM:
    """An interpolated add-k 4-gram language model.

    Counts are stored per order as ``{context_tuple: {word: count}}``. Scoring
    walks down from the top order, mixing each order's MLE by INTERP_WEIGHTS and
    letting the add-k unigram catch anything unseen. It's not the fastest design,
    but it's transparent — you can read the smoothing straight off the code.
    """

    def __init__(self, order: int = NGRAM_ORDER, add_k: float = ADD_K):
        if order < 1:
            raise ValueError("n-gram order must be at least 1.")
        self.order = order
        self.add_k = add_k
        # counts[o] maps a context of length o to a Counter-like {word: count}.
        self.counts: list[dict[tuple[str, ...], dict[str, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(order)
        ]
        self.vocab: set[str] = set()
        self._total_unigrams = 0

    def _pad(self, tokens: list[str]) -> list[str]:
        return [BOS] * (self.order - 1) + tokens + [EOS]

    def fit(self, sentences: Iterable[list[str]]) -> NgramLM:
        """Accumulate n-gram counts from already-tokenised sentences."""
        for tokens in sentences:
            padded = self._pad(tokens)
            self.vocab.update(tokens)
            self.vocab.add(EOS)
            for i in range(self.order - 1, len(padded)):
                word = padded[i]
                for o in range(self.order):
                    context = tuple(padded[i - o:i]) if o else ()
                    self.counts[o][context][word] += 1
        self._total_unigrams = sum(self.counts[0][()].values())
        logger.info("4-gram LM: vocab=%d, unigram tokens=%d",
                    len(self.vocab), self._total_unigrams)
        return self

    def _order_prob(self, o: int, context: tuple[str, ...], word: str) -> float:
        """Maximum-likelihood probability at one order. Zero if context unseen."""
        ctx_counts = self.counts[o].get(context)
        if not ctx_counts:
            return 0.0
        total = sum(ctx_counts.values())
        return ctx_counts.get(word, 0) / total if total else 0.0

    def _unigram_prob(self, word: str) -> float:
        """add-k smoothed unigram, so nothing is ever exactly zero."""
        v = max(1, len(self.vocab))
        c = self.counts[0][()].get(word, 0)
        return (c + self.add_k) / (self._total_unigrams + self.add_k * v)

    def word_logprob(self, context: tuple[str, ...], word: str) -> float:
        """Interpolated log-probability of ``word`` after ``context``.

        We blend orders top-down by INTERP_WEIGHTS, then add the add-k unigram as
        the final backstop so the total probability mass is always positive.
        """
        prob = 0.0
        # Orders from top (order-1 of context) down to bigram.
        for depth, weight in enumerate(INTERP_WEIGHTS[: self.order - 1]):
            o = self.order - 1 - depth
            ctx = context[-o:] if o else ()
            prob += weight * self._order_prob(o, ctx, word)
        # Remaining weight goes to the smoothed unigram.
        used = sum(INTERP_WEIGHTS[: self.order - 1])
        prob += (1.0 - used) * self._unigram_prob(word)
        # Floor guards against a degenerate all-zero blend (shouldn't happen
        # given the add-k unigram, but keeps the log finite no matter what).
        return math.log(max(prob, 1e-12))

    def sentence_logprob(self, tokens: list[str]) -> float:
        """Total log-probability of a sentence under the model."""
        padded = self._pad(tokens)
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            context = tuple(padded[i - (self.order - 1):i])
            total += self.word_logprob(context, padded[i])
        return total

    def normalized_logprob(self, sentence: str) -> float:
        """Length-normalised log-prob — the score we compare across a pair.

        We divide by the number of scored positions (words plus the EOS) so long
        and short sentences are comparable, the same spirit as SLOR's /|s|.
        """
        tokens = _tokenize(sentence)
        n = len(tokens) + 1  # +1 for EOS
        return self.sentence_logprob(tokens) / max(1, n)


def _read_corpus_sentences(corpus_path: Path) -> Iterator[list[str]]:
    """Yield tokenised sentences from a dose corpus (CoNLL-U) or plain text."""
    corpus_path = Path(corpus_path)
    if corpus_path.suffix == ".conllu":
        from drc.data.parse import read_metadata

        for sent in read_metadata(corpus_path):
            yield _tokenize(sent.text)
    else:
        with open(corpus_path, encoding="utf-8") as fh:
            for line in fh:
                _, sep, text = line.partition("\t")
                yield _tokenize(text if sep else line)


def train_ngram_lm(corpus_path: Path, order: int = NGRAM_ORDER) -> NgramLM:
    """Train a 4-gram LM on one dose corpus.

    Tries KenLM only if explicitly available *and* useful; we ship the
    pure-Python model as the default because it has zero install footprint and
    the baseline doesn't need to be fast. KenLM scoring would also require
    building ARPA files, which isn't worth the complexity here — so we note the
    option and move on.
    """
    logger.info("Training %d-gram LM on %s", order, corpus_path.name)
    lm = NgramLM(order=order)
    lm.fit(_read_corpus_sentences(corpus_path))
    return lm


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
    """Standard error of a proportion: sqrt(p(1-p)/n)."""
    if n == 0:
        return 0.0
    return math.sqrt(accuracy * (1.0 - accuracy) / n)


def evaluate_pair_set(lm: NgramLM, items: list[EvalItem]) -> tuple[int, int]:
    """Score the LM on one construction's pairs. Returns (n_pairs, n_correct)."""
    n_correct = 0
    for item in items:
        good = lm.normalized_logprob(item.good_sentence)
        bad = lm.normalized_logprob(item.bad_sentence)
        if good > bad:
            n_correct += 1
    return len(items), n_correct


def _load_existing_keys(csv_path: Path) -> set[tuple[str, str, str, str]]:
    """Keys already scored, so a resumed run skips them."""
    done: set[tuple[str, str, str, str]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            done.add(
                (row["model_construction"], row["dose"],
                 row["seed"], row["eval_construction"])
            )
    logger.info("Resuming: %d rows already in %s.", len(done), csv_path.name)
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


def _dose_corpus_path(config: dict[str, Any], config_path: Path,
                      construction: str, dose: str) -> Path:
    """Where the dose corpus lives (matches dose_corpora.py naming)."""
    root = resolve_path(config_path, config["paths"]["dose_corpora"])
    return root / f"{construction}_dose-{dose}.conllu"


def run(
    config_path: Path,
    only_construction: str | None = None,
) -> None:
    """Train an n-gram LM per dose corpus and score it on all four test sets.

    The n-gram LM has no random seed — it's a deterministic count over the
    corpus — but the neural sweep records three seeds per (construction, dose).
    To keep the CSV schema aligned for downstream joins, we emit one row per
    seed with identical numbers. That way the baseline lines up against the
    neural results without special-casing the join.
    """
    config = load_config(config_path)
    items_root = resolve_path(config_path, config["paths"]["eval_items"])
    results_root = resolve_path(config_path, config["paths"]["results"])
    csv_path = results_root / "ngram_baseline.csv"

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

    constructions = (only_construction,) if only_construction else CONSTRUCTIONS
    done = _load_existing_keys(csv_path)

    for model_construction in constructions:
        for dose in DOSE_LEVELS:
            dose = str(dose)
            # The LM is the same across seeds, so train it once per dose corpus.
            seeds_pending = [
                s for s in SEEDS
                if any(
                    (model_construction, dose, str(s), c) not in done
                    for c in eval_sets
                )
            ]
            if not seeds_pending:
                continue

            corpus = _dose_corpus_path(
                config, config_path, model_construction, dose
            )
            if not corpus.exists():
                logger.warning("Dose corpus %s missing; skipping.", corpus)
                continue

            lm = train_ngram_lm(corpus)

            # Score once, then write the identical numbers under each seed.
            scored: dict[str, tuple[int, int]] = {}
            for eval_construction, items in eval_sets.items():
                scored[eval_construction] = evaluate_pair_set(lm, items)

            for seed in SEEDS:
                for eval_construction, (n_pairs, n_correct) in scored.items():
                    key = (model_construction, dose, str(seed), eval_construction)
                    if key in done:
                        continue
                    accuracy = n_correct / n_pairs if n_pairs else 0.0
                    row = {
                        "model_construction": model_construction,
                        "dose": dose,
                        "seed": seed,
                        "eval_construction": eval_construction,
                        "n_pairs": n_pairs,
                        "n_correct": n_correct,
                        "accuracy": round(accuracy, 6),
                        "std_error": round(
                            _binomial_std_error(accuracy, n_pairs), 6
                        ),
                    }
                    _append_row(csv_path, row)
            logger.info("Scored %s dose=%s: %s",
                        model_construction, dose,
                        {c: f"{nc}/{n}" for c, (n, nc) in scored.items()})

    logger.info("n-gram baseline complete. Results in %s", csv_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base config with the `paths` section.",
    )
    parser.add_argument(
        "--construction", choices=CONSTRUCTIONS, default=None,
        help="Run only this construction's dose corpora (default: all four).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, only_construction=args.construction)


if __name__ == "__main__":
    main()
