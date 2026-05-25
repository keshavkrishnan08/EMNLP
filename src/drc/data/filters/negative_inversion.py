"""Negative inversion: fronted negative adverbial + subject-aux inversion.

When a negative or restrictive adverbial is fronted, English obligatorily flips
the subject and the auxiliary, just like in a question:

    "never have I seen such a mess"
     ^^^^^ ^^^^ ^
     NEG   AUX  subject

Drop the inversion and the sentence breaks: *"never I have seen ...". The trigger
is a closed class of negative/restrictive expressions — "never", "rarely",
"hardly", "not only", "no sooner", "little" (in its negative sense), and friends.
We match a sentence opening with one of these, immediately followed by an
auxiliary that lands *before* the subject. We check that the token after the
trigger is an AUX (or a form of BE/DO/HAVE) and that a subject follows it, which
is the inversion's signature.

Reference: Haegeman (2000), "Negative preposing, negative inversion, and the
split CP," in *Negation and Polarity*.
"""

from __future__ import annotations

from .base import FilterMatch, words

# Single-word triggers that front with obligatory inversion. These are negative
# ("never", "nowhere") or restrictive/scalar ("rarely", "seldom", "hardly",
# "scarcely", "barely", "little") adverbs that license subject-aux inversion when
# they open the clause.
_SINGLE_TRIGGERS = frozenset(
    {
        "never",
        "rarely",
        "seldom",
        "hardly",
        "scarcely",
        "barely",
        "nowhere",
        "little",
    }
)

# Multi-word triggers, matched on the lowercased opening tokens. "no sooner",
# "not only", and "not until" all front as a unit and force inversion.
_MULTI_TRIGGERS = (
    ("no", "sooner"),
    ("not", "only"),
    ("not", "until"),
)

# Auxiliary lemmas that can invert past the subject. Stanza tags these AUX, but
# the lemma check keeps us robust when a parse marks "do"/"have" as VERB.
_AUX_LEMMAS = frozenset({"be", "do", "have", "will", "would", "can", "could",
                         "shall", "should", "may", "might", "must"})


def _is_aux(word) -> bool:
    """True if a word can serve as the inverting auxiliary."""
    return word.upos == "AUX" or word.lemma in _AUX_LEMMAS


def _is_subject(word) -> bool:
    """True if a word looks like the post-auxiliary subject."""
    return word.deprel in {"nsubj", "nsubj:pass"} or word.upos in {"PRON", "PROPN", "NOUN"}


class NegativeInversionFilter:
    """Detect a fronted negative adverbial followed by subject-aux inversion.

    We match the trigger at the left edge, then require an auxiliary in the very
    next slot (after the trigger's one or two words) with a subject right behind
    it. The AUX-before-subject ordering is what marks the inversion — without it
    the fronted adverbial is just an ordinary sentence opener.
    """

    code = "negative_inversion"

    def __call__(self, sentence) -> FilterMatch:
        ws = words(sentence)
        if len(ws) < 3:
            return FilterMatch.miss(note="too short for trigger + AUX + subject")

        # Figure out where the trigger ends. Try a two-word trigger first, then
        # fall back to a single-word one. ``trig_end`` is the index of the first
        # token *after* the trigger.
        trig_end = None
        first, second = ws[0].text.lower(), ws[1].text.lower()
        if (first, second) in _MULTI_TRIGGERS:
            trig_end = 2
        elif first in _SINGLE_TRIGGERS:
            trig_end = 1

        if trig_end is None:
            return FilterMatch.miss(note="no clause-initial negative trigger")

        # Need an AUX and then a subject after the trigger.
        if trig_end + 1 >= len(ws):
            return FilterMatch.miss(note="trigger not followed by AUX + subject")

        aux, subj = ws[trig_end], ws[trig_end + 1]

        # The auxiliary must sit immediately after the fronted adverbial...
        if not _is_aux(aux):
            return FilterMatch.miss(note="no auxiliary right after the trigger")

        # ...and the subject must follow the auxiliary, not precede it. That
        # AUX-before-subject order is the whole construction.
        if not _is_subject(subj):
            return FilterMatch.miss(note="no subject after the inverted auxiliary")

        trigger_txt = " ".join(w.text for w in ws[:trig_end])
        note = f"{trigger_txt!r} + {aux.text}(AUX) + {subj.text}(subj) inverted"
        return FilterMatch.hit(0, trig_end + 2, note=note)


# Module-level instance so the package init can register it directly.
negative_inversion = NegativeInversionFilter()
