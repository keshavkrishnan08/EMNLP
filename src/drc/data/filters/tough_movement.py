"""Tough-movement: "this book is tough to read".

A small class of adjectives ("tough", "easy", "hard"...) lets the object of an
embedded verb surface as the matrix subject:

    "the painting is easy to admire"
     ^^^^^^^^^^^    ^^ ^^^^    ^^^^^
     subject       BE  ADJ to  VERB

The subject ("the painting") is understood as what gets admired — there's a gap
after the embedded verb. This is a textbook poverty-of-stimulus case: the
dependency is long-distance yet acquired early. We don't try to verify the gap
syntactically (that's brittle); instead we match the reliable surface frame
BE + tough-adjective + "to" + VERB, which the tough-class lexicon makes precise.

Reference: Hu et al. (2020), "A Systematic Assessment of Syntactic
Generalization in Neural Language Models" (SyntaxGym).
"""

from __future__ import annotations

from .base import FilterMatch, words

# The adjectives that license tough-movement. It's a closed-ish class — these
# are the ones that take a "to"-infinitive with a subject-as-object reading.
# Lemmas, so inflection ("easier") collapses to the base form.
TOUGH_ADJECTIVES = frozenset(
    {
        "tough",
        "easy",
        "hard",
        "difficult",
        "simple",
        "fun",
        "impossible",
        "tricky",
        "tedious",
        "pleasant",
        "awkward",
        "painful",
        "enjoyable",
        "boring",
        "exciting",
        "comfortable",
        "uncomfortable",
        "convenient",
        "inconvenient",
        "cheap",
        "expensive",
        "safe",
        "dangerous",
    }
)


class ToughMovementFilter:
    """Detect the BE + tough-ADJ + to + VERB frame.

    We anchor on the copula's lemma ("is"/"was"/"been" all collapse to "be"),
    then walk the next three slots. The tough-adjective lexicon does the heavy
    lifting: it's what separates true tough-movement from ordinary "is ready to
    leave" predicates, which aren't in the class.
    """

    code = "tough_movement"

    def __call__(self, sentence) -> FilterMatch:
        ws = words(sentence)

        # Need four tokens from the copula onward: be + ADJ + to + VERB.
        for i in range(len(ws) - 3):
            be, adj, to, verb = ws[i], ws[i + 1], ws[i + 2], ws[i + 3]

            # Slot 0: a form of "be". Lemma check covers is/are/was/were/been.
            if be.lemma != "be":
                continue

            # Slot 1: the tough-class adjective. Lemma keeps "easier"/"easy"
            # together; membership in the lexicon is the decisive test.
            if adj.lemma not in TOUGH_ADJECTIVES:
                continue

            # Slot 2: the infinitival "to". Lemma "to" sidesteps any odd casing.
            if to.lemma != "to":
                continue

            # Slot 3: the embedded verb whose object is the matrix subject.
            if verb.upos != "VERB":
                continue

            note = f"be+{adj.text}+to+{verb.text} (tough frame)"
            return FilterMatch.hit(i, i + 4, note=note)

        return FilterMatch.miss(note="no BE+tough-ADJ+to+VERB frame")


tough_movement = ToughMovementFilter()
