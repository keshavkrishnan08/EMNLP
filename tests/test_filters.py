"""Unit tests for the four construction filters.

Each filter gets a hand-built fake parse that should match and one that
shouldn't. Rows are ``(text, upos, lemma, feats, deprel, head)`` and word ids
run 1..N in order (see ``make_sentence`` in conftest). The ``make_sentence``
helper arrives via the ``make_sentence_fixture`` pytest fixture.
"""

from __future__ import annotations

from drc.data.filters import get_filter

# --- AANN: a/an + ADJ + NUM + plural NOUN ---------------------------------

def test_aann_matches_a_lovely_three_days(make_sentence_fixture):
    """'a lovely three days' is the canonical AANN frame."""
    aann = get_filter("aann")
    # The plural noun (id 4) is the head; the article (id 1) attaches to it.
    sent = make_sentence_fixture(
        [
            ("a", "DET", "a", "Definite=Ind|PronType=Art", "det", 4),
            ("lovely", "ADJ", "lovely", "Degree=Pos", "amod", 4),
            ("three", "NUM", "three", "NumType=Card", "nummod", 4),
            ("days", "NOUN", "day", "Number=Plur", "root", 0),
        ]
    )
    result = aann(sent)
    assert bool(result)
    assert result.span == (0, 4)


def test_aann_rejects_singular_noun(make_sentence_fixture):
    """'a lovely day' is an ordinary NP, not AANN — singular and no numeral."""
    aann = get_filter("aann")
    sent = make_sentence_fixture(
        [
            ("a", "DET", "a", "Definite=Ind|PronType=Art", "det", 3),
            ("lovely", "ADJ", "lovely", "Degree=Pos", "amod", 3),
            ("day", "NOUN", "day", "Number=Sing", "root", 0),
        ]
    )
    assert not bool(aann(sent))


# --- Comparative correlative: "the X-er, the Y-er" ------------------------

def test_comparative_correlative_matches_paired_clauses(make_sentence_fixture):
    """'the harder you try, the better it gets' is the classic paired frame."""
    cc = get_filter("comparative_correlative")
    text = "the harder you try , the better it gets"
    sent = make_sentence_fixture(
        [
            ("the", "DET", "the", "Definite=Def|PronType=Art", "det", 2),
            ("harder", "ADV", "hard", "Degree=Cmp", "advmod", 4),
            ("you", "PRON", "you", "Person=2", "nsubj", 4),
            ("try", "VERB", "try", "VerbForm=Fin", "advcl", 8),
            (",", "PUNCT", ",", None, "punct", 8),
            ("the", "DET", "the", "Definite=Def|PronType=Art", "det", 7),
            ("better", "ADV", "well", "Degree=Cmp", "advmod", 8),
            ("it", "PRON", "it", "Person=3", "nsubj", 8),
            ("gets", "VERB", "get", "VerbForm=Fin", "root", 0),
        ],
        text=text,
    )
    assert bool(cc(sent))


def test_comparative_correlative_rejects_plain_sentence(make_sentence_fixture):
    """'she tried hard and it got better' has no paired the+comparative frame."""
    cc = get_filter("comparative_correlative")
    text = "she tried hard and it got better"
    sent = make_sentence_fixture(
        [
            ("she", "PRON", "she", "Person=3", "nsubj", 2),
            ("tried", "VERB", "try", "VerbForm=Fin", "root", 0),
            ("hard", "ADV", "hard", "Degree=Pos", "advmod", 2),
            ("and", "CCONJ", "and", None, "cc", 6),
            ("it", "PRON", "it", "Person=3", "nsubj", 6),
            ("got", "VERB", "get", "VerbForm=Fin", "conj", 2),
            ("better", "ADV", "well", "Degree=Cmp", "advmod", 6),
        ],
        text=text,
    )
    assert not bool(cc(sent))


# --- Tough movement: BE + tough-ADJ + to + VERB ---------------------------

def test_tough_movement_matches_easy_to_read(make_sentence_fixture):
    """'this book is easy to read' is the BE + tough-ADJ + to + VERB frame."""
    tough = get_filter("tough_movement")
    sent = make_sentence_fixture(
        [
            ("this", "DET", "this", "PronType=Dem", "det", 2),
            ("book", "NOUN", "book", "Number=Sing", "nsubj", 4),
            ("is", "AUX", "be", "VerbForm=Fin", "cop", 4),
            ("easy", "ADJ", "easy", "Degree=Pos", "root", 0),
            ("to", "PART", "to", None, "mark", 6),
            ("read", "VERB", "read", "VerbForm=Inf", "xcomp", 4),
        ]
    )
    result = tough(sent)
    assert bool(result)
    # The frame starts at the copula (id 3 -> index 2) and spans four tokens.
    assert result.span == (2, 6)


def test_tough_movement_rejects_eager_adjective(make_sentence_fixture):
    """'this book is eager to please' — 'eager' isn't in the tough class."""
    tough = get_filter("tough_movement")
    sent = make_sentence_fixture(
        [
            ("this", "DET", "this", "PronType=Dem", "det", 2),
            ("book", "NOUN", "book", "Number=Sing", "nsubj", 4),
            ("is", "AUX", "be", "VerbForm=Fin", "cop", 4),
            ("eager", "ADJ", "eager", "Degree=Pos", "root", 0),
            ("to", "PART", "to", None, "mark", 6),
            ("please", "VERB", "please", "VerbForm=Inf", "xcomp", 4),
        ]
    )
    assert not bool(tough(sent))


# --- Resultative: V governing both an obj and an adjectival xcomp ---------

def test_resultative_matches_hammered_metal_flat(make_sentence_fixture):
    """'he hammered the metal flat': verb with obj + adjectival xcomp."""
    resultative = get_filter("resultative")
    sent = make_sentence_fixture(
        [
            ("he", "PRON", "he", "Person=3", "nsubj", 2),
            ("hammered", "VERB", "hammer", "VerbForm=Fin", "root", 0),
            ("the", "DET", "the", "Definite=Def", "det", 4),
            ("metal", "NOUN", "metal", "Number=Sing", "obj", 2),
            ("flat", "ADJ", "flat", "Degree=Pos", "xcomp", 2),
        ]
    )
    result = resultative(sent)
    assert bool(result)
    # Span runs from the verb (id 2 -> index 1) through the result adjective (id 5).
    assert result.span == (1, 5)


def test_resultative_rejects_verb_with_only_object(make_sentence_fixture):
    """'he hammered the metal' has an obj but no result AP — not resultative."""
    resultative = get_filter("resultative")
    sent = make_sentence_fixture(
        [
            ("he", "PRON", "he", "Person=3", "nsubj", 2),
            ("hammered", "VERB", "hammer", "VerbForm=Fin", "root", 0),
            ("the", "DET", "the", "Definite=Def", "det", 4),
            ("metal", "NOUN", "metal", "Number=Sing", "obj", 2),
        ]
    )
    assert not bool(resultative(sent))
