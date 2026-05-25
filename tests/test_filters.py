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


# --- Existential there: expletive there + BE + nominal pivot ---------------

def test_existential_there_matches_there_are_three_cats(make_sentence_fixture):
    """'there are three cats' is the expletive-there + BE + pivot frame."""
    et = get_filter("existential_there")
    sent = make_sentence_fixture(
        [
            ("there", "PRON", "there", None, "expl", 2),
            ("are", "AUX", "be", "Number=Plur|VerbForm=Fin", "root", 0),
            ("three", "NUM", "three", "NumType=Card", "nummod", 4),
            ("cats", "NOUN", "cat", "Number=Plur", "nsubj", 2),
            ("on", "ADP", "on", None, "case", 7),
            ("the", "DET", "the", "Definite=Def", "det", 7),
            ("mat", "NOUN", "mat", "Number=Sing", "obl", 2),
        ]
    )
    result = et(sent)
    assert bool(result)
    # there (idx 0) + be (idx 1) + pivot head 'three' (idx 2): a three-token frame.
    assert result.span == (0, 3)


def test_existential_there_rejects_locative_there(make_sentence_fixture):
    """'put it there' — sentence-final locative 'there', no existential frame."""
    et = get_filter("existential_there")
    sent = make_sentence_fixture(
        [
            ("put", "VERB", "put", "VerbForm=Fin", "root", 0),
            ("it", "PRON", "it", "Person=3", "obj", 1),
            ("there", "ADV", "there", None, "advmod", 1),
        ]
    )
    assert not bool(et(sent))


# --- It-cleft: it + BE + focus XP + relativizer ----------------------------

def test_it_cleft_matches_it_was_the_dog_that_barked(make_sentence_fixture):
    """'it was the dog that barked' is the canonical it-cleft."""
    cleft = get_filter("it_cleft")
    sent = make_sentence_fixture(
        [
            ("it", "PRON", "it", "Person=3", "nsubj", 4),
            ("was", "AUX", "be", "VerbForm=Fin", "cop", 4),
            ("the", "DET", "the", "Definite=Def", "det", 4),
            ("dog", "NOUN", "dog", "Number=Sing", "root", 0),
            ("that", "PRON", "that", "PronType=Rel", "nsubj", 6),
            ("barked", "VERB", "bark", "VerbForm=Fin", "acl:relcl", 4),
        ]
    )
    result = cleft(sent)
    assert bool(result)
    # Spans from clause-initial 'it' (idx 0) through the relativizer 'that' (idx 4).
    assert result.span == (0, 5)


def test_it_cleft_rejects_subject_aux_question(make_sentence_fixture):
    """'was it the dog that barked' is a question — 'it' isn't clause-initial."""
    cleft = get_filter("it_cleft")
    sent = make_sentence_fixture(
        [
            ("was", "AUX", "be", "VerbForm=Fin", "cop", 4),
            ("it", "PRON", "it", "Person=3", "nsubj", 4),
            ("the", "DET", "the", "Definite=Def", "det", 4),
            ("dog", "NOUN", "dog", "Number=Sing", "root", 0),
            ("that", "PRON", "that", "PronType=Rel", "nsubj", 6),
            ("barked", "VERB", "bark", "VerbForm=Fin", "acl:relcl", 4),
        ]
    )
    assert not bool(cleft(sent))


# --- Negative inversion: fronted negative adverbial + subject-aux inversion -

def test_negative_inversion_matches_never_have_I_seen(make_sentence_fixture):
    """'never have I seen ...' fronts 'never' and inverts subject and aux."""
    neg = get_filter("negative_inversion")
    sent = make_sentence_fixture(
        [
            ("never", "ADV", "never", None, "advmod", 4),
            ("have", "AUX", "have", "VerbForm=Fin", "aux", 4),
            ("I", "PRON", "I", "Person=1", "nsubj", 4),
            ("seen", "VERB", "see", "VerbForm=Part", "root", 0),
            ("such", "DET", "such", None, "det", 6),
            ("messes", "NOUN", "mess", "Number=Plur", "obj", 4),
        ]
    )
    result = neg(sent)
    assert bool(result)
    # 'never' (idx 0) + aux 'have' (idx 1) + subject 'I' (idx 2): a three-token frame.
    assert result.span == (0, 3)


def test_negative_inversion_rejects_uninverted_order(make_sentence_fixture):
    """'never I have seen ...' keeps the subject before the aux — not inverted."""
    neg = get_filter("negative_inversion")
    sent = make_sentence_fixture(
        [
            ("never", "ADV", "never", None, "advmod", 4),
            ("I", "PRON", "I", "Person=1", "nsubj", 4),
            ("have", "AUX", "have", "VerbForm=Fin", "aux", 4),
            ("seen", "VERB", "see", "VerbForm=Part", "root", 0),
            ("such", "DET", "such", None, "det", 6),
            ("messes", "NOUN", "mess", "Number=Plur", "obj", 4),
        ]
    )
    assert not bool(neg(sent))


# --- Subject-to-object raising: ECM verb + NP + to + VERB ------------------

def test_subject_to_object_raising_matches_believed_him_to_be(make_sentence_fixture):
    """'they believed him to be guilty' is the ECM / raising-to-object frame."""
    raising = get_filter("subject_to_object_raising")
    sent = make_sentence_fixture(
        [
            ("they", "PRON", "they", "Person=3", "nsubj", 2),
            ("believed", "VERB", "believe", "VerbForm=Fin", "root", 0),
            ("him", "PRON", "he", "Case=Acc", "obj", 2),
            ("to", "PART", "to", None, "mark", 5),
            ("be", "AUX", "be", "VerbForm=Inf", "xcomp", 2),
            ("guilty", "ADJ", "guilty", "Degree=Pos", "xcomp", 2),
        ]
    )
    result = raising(sent)
    assert bool(result)
    # verb 'believed' (idx 1) through embedded 'be' (idx 4).
    assert result.span == (1, 5)


def test_subject_to_object_raising_rejects_missing_to(make_sentence_fixture):
    """'they believed him be guilty' drops the infinitival 'to' — ungrammatical."""
    raising = get_filter("subject_to_object_raising")
    sent = make_sentence_fixture(
        [
            ("they", "PRON", "they", "Person=3", "nsubj", 2),
            ("believed", "VERB", "believe", "VerbForm=Fin", "root", 0),
            ("him", "PRON", "he", "Case=Acc", "obj", 2),
            ("be", "AUX", "be", "VerbForm=Inf", "xcomp", 2),
            ("guilty", "ADJ", "guilty", "Degree=Pos", "xcomp", 2),
        ]
    )
    assert not bool(raising(sent))
