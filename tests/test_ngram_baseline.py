"""Tests for the pure-Python interpolated 4-gram baseline LM.

No torch needed — this model is plain counting and arithmetic.
"""

from __future__ import annotations

from drc.eval.ngram_baseline import NgramLM, _tokenize

# A tiny toy corpus with a clear, repeated word order. The LM should learn that
# "the cat sat on the mat" style sequences are likely and reward them.
TOY_CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran in the park",
    "the cat ran in the park",
    "a cat sat on the mat",
    "the cat sat quietly on the mat",
]


def _trained_lm() -> NgramLM:
    lm = NgramLM()
    lm.fit(_tokenize(s) for s in TOY_CORPUS)
    return lm


def test_lm_learns_vocab_and_counts():
    lm = _trained_lm()
    assert "cat" in lm.vocab
    assert "mat" in lm.vocab
    assert lm._total_unigrams > 0


def test_grammatical_sequence_outranks_scrambled():
    """A seen, well-formed sentence should beat a scrambled word salad."""
    lm = _trained_lm()
    grammatical = lm.normalized_logprob("the cat sat on the mat")
    scrambled = lm.normalized_logprob("mat the on sat cat the")
    assert grammatical > scrambled


def test_unseen_words_get_finite_score():
    """The add-k unigram floor keeps OOV sentences from going to -inf."""
    lm = _trained_lm()
    score = lm.normalized_logprob("zzz qqq wxyz")
    assert score == score  # not NaN
    assert score > float("-inf")


def test_higher_order_context_rewards_seen_continuation():
    """A continuation seen in the corpus should out-score a never-seen one."""
    lm = _trained_lm()
    # "sat on the" -> "mat" appears repeatedly; "sat on the park" never does.
    seen = lm.normalized_logprob("the cat sat on the mat")
    unseen = lm.normalized_logprob("the cat sat on the park")
    assert seen > unseen
