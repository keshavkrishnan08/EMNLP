"""SLOR scoring for masked language models.

SLOR (Syntactic Log-Odds Ratio, Pauls & Klein 2012; Lau et al. 2017) corrects a
sentence's log-probability for the fact that rare words drag it down regardless
of grammaticality. You subtract off what a unigram model would have assigned,
then divide by length so long and short sentences land on the same scale:

    SLOR(s) = (log P(s) - log P_unigram(s)) / |s|

For a masked LM there's no left-to-right log P(s), so we use the
*pseudo*-log-likelihood (Salazar et al. 2020): mask each token in turn, ask the
model for the log-probability of the true token given everything else, and sum.
That's |s| forward passes per sentence, which is the slow part — so we batch
them, stacking one masked copy per position into a single forward call.

Everything torch/transformers gets imported lazily inside the functions, so the
module imports fine in a bare environment (handy for unit-testing the unigram
math without a GPU around).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Batch size for the per-position masked forward passes. A sentence of length n
# becomes n masked copies; we run them in chunks this big so a long sentence
# doesn't blow up memory. Tune down if you hit OOM on a small GPU.
DEFAULT_MASK_BATCH_SIZE = 64

# Floor for unigram probabilities so an out-of-vocabulary word costs a large but
# finite penalty instead of -inf, which would make SLOR undefined.
_OOV_FLOOR_COUNT = 1

# Crude word tokeniser for the unigram side. We lowercase and keep word-ish runs
# only — the unigram baseline is a normaliser, not a model, so it doesn't need
# the LM's subword vocabulary. Matching that vocabulary would actually defeat
# the purpose: we want a corpus-frequency correction, not a second LM.
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _words(sentence: str) -> list[str]:
    """Lowercase word tokens for the unigram model."""
    return _WORD_RE.findall(sentence.lower())


def build_unigram_counts(corpus_path: Path) -> Counter[str]:
    """Count word frequencies over a training corpus for the unigram correction.

    Accepts either a CoNLL-U dose corpus (we pull the ``# text =`` lines) or a
    plain-text file (one sentence per line). We sniff the format by extension so
    callers don't have to care which they hand us.

    The returned Counter is what ``unigram_logprob`` consumes. Build it once per
    dose corpus and reuse it across every sentence you score.
    """
    corpus_path = Path(corpus_path)
    counts: Counter[str] = Counter()
    if corpus_path.suffix == ".conllu":
        # Reuse the parser's metadata reader so we read the exact same text the
        # model trained on, rather than re-deriving it from token rows.
        from drc.data.parse import read_metadata

        for sent in read_metadata(corpus_path):
            counts.update(_words(sent.text))
    else:
        with open(corpus_path, encoding="utf-8") as fh:
            for line in fh:
                # Tolerate the download.py "domain<TAB>text" layout too.
                _, sep, text = line.partition("\t")
                counts.update(_words(text if sep else line))
    logger.info("Unigram model: %d types, %d tokens from %s",
                len(counts), sum(counts.values()), corpus_path.name)
    return counts


def unigram_logprob(sentence: str, unigram_counts: Counter[str]) -> float:
    """Log P of a sentence under the corpus unigram model.

    Probabilities come straight from relative frequency, with a count floor so
    unseen words get a finite (if harsh) penalty. We score the same word tokens
    SLOR will divide by, so the units line up.
    """
    total = sum(unigram_counts.values())
    if total == 0:
        raise ValueError("Unigram counts are empty; build them before scoring.")
    logp = 0.0
    for w in _words(sentence):
        c = unigram_counts.get(w, 0) or _OOV_FLOOR_COUNT
        logp += math.log(c / total)
    return logp


def _token_count(sentence: str) -> int:
    """Length used to normalise SLOR. Word tokens, never zero."""
    return max(1, len(_words(sentence)))


def pseudo_log_likelihood(
    model: Any,
    tokenizer: Any,
    sentence: str,
    *,
    mask_batch_size: int = DEFAULT_MASK_BATCH_SIZE,
    device: str | None = None,
) -> float:
    """Masked-LM pseudo-log-likelihood: sum of log P(token | rest) over tokens.

    The recipe (Salazar et al. 2020): for each non-special token position, mask
    that one position, run the model, and read off the log-probability the model
    assigns to the token that was actually there. Sum across positions.

    We build all the masked copies up front and push them through in batches,
    which turns |s| separate forward passes into a handful. Special tokens (CLS,
    SEP, pad) are never masked and never scored — only real content positions
    contribute.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(
            "Tokenizer has no mask token; SLOR needs a masked LM tokenizer."
        )

    enc = tokenizer(sentence, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"][0]
    attention = enc.get("attention_mask")
    attention = attention[0] if attention is not None else torch.ones_like(input_ids)

    # Score every position that isn't a special token. get_special_tokens_mask
    # marks CLS/SEP/pad with 1; we want the 0s.
    special = tokenizer.get_special_tokens_mask(
        input_ids.tolist(), already_has_special_tokens=True
    )
    score_positions = [i for i, s in enumerate(special) if s == 0]
    if not score_positions:
        return 0.0

    # One masked copy of the sentence per scored position.
    n = len(score_positions)
    batch = input_ids.unsqueeze(0).repeat(n, 1).clone()
    for row, pos in enumerate(score_positions):
        batch[row, pos] = mask_id
    attn = attention.unsqueeze(0).repeat(n, 1)

    target_ids = input_ids[score_positions]
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, n, mask_batch_size):
            stop = min(start + mask_batch_size, n)
            chunk = batch[start:stop].to(device)
            chunk_attn = attn[start:stop].to(device)
            logits = model(input_ids=chunk, attention_mask=chunk_attn).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            # For row r in this chunk, the masked position is score_positions[start+r].
            rows = torch.arange(stop - start)
            positions = torch.tensor(
                [score_positions[start + r] for r in range(stop - start)]
            )
            targets = target_ids[start:stop].to(device)
            picked = log_probs[rows, positions, targets]
            total += float(picked.sum().item())
    return total


def slor(
    model: Any,
    tokenizer: Any,
    sentence: str,
    unigram_counts: Counter[str],
    *,
    mask_batch_size: int = DEFAULT_MASK_BATCH_SIZE,
    device: str | None = None,
) -> float:
    """Full SLOR score for one sentence: PLL minus unigram, over length.

    Higher means "more acceptable to the model after correcting for word
    rarity." The minimal-pair test then just checks whether the good sentence
    out-scores the bad one.
    """
    pll = pseudo_log_likelihood(
        model, tokenizer, sentence,
        mask_batch_size=mask_batch_size, device=device,
    )
    uni = unigram_logprob(sentence, unigram_counts)
    return (pll - uni) / _token_count(sentence)


def slor_batch(
    model: Any,
    tokenizer: Any,
    sentences: Iterable[str],
    unigram_counts: Counter[str],
    *,
    mask_batch_size: int = DEFAULT_MASK_BATCH_SIZE,
    device: str | None = None,
) -> list[float]:
    """Convenience wrapper: SLOR over many sentences, returned in order."""
    return [
        slor(model, tokenizer, s, unigram_counts,
             mask_batch_size=mask_batch_size, device=device)
        for s in sentences
    ]
