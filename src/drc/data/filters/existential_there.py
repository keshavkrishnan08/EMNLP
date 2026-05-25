"""Existential "there": "there is/are NP".

English has a dedicated construction where an unstressed expletive "there" fills
the subject slot and the logical subject — the "pivot" — shows up after the
copula:

    "there are three cats on the mat"
     ^^^^^ ^^^ ^^^^^^^^^^
     expl  BE  pivot NP

"There" here isn't the locative adverb ("over there"); it's a semantically empty
placeholder that lets a sentence assert existence. The interesting agreement
twist is that the verb agrees with the post-copular pivot, not with "there", so
"there *are* three cats" but "there *is* one cat". We match the surface frame
expletive-there + form of BE + a nominal pivot head, and use the dependency
relation ``expl`` when it's available to separate it from locative "there".

Reference: McNally (2011), "Existential Sentences," in *Semantics: An
International Handbook of Natural Language Meaning*.
"""

from __future__ import annotations

from .base import FilterMatch, words

# Categories that can head the post-copular pivot. A bare numeral ("there are
# three") or determiner ("there is no time") can stand in for the noun, so we
# accept the usual nominal heads rather than insisting on NOUN.
_PIVOT_UPOS = frozenset({"NOUN", "PROPN", "PRON", "NUM", "DET"})


class ExistentialThereFilter:
    """Detect expletive "there" + a form of BE + a nominal pivot.

    We anchor on the expletive. Stanza tags it ``expl`` when the parse is clean,
    which is the surest signal it's the existential "there" and not the locative
    one. We fall back to a clause-initial "there" tagged PRON/DET/ADV when the
    deprel is missing, then confirm the next token is a form of "be" and the one
    after that can head a noun phrase.
    """

    code = "existential_there"

    def __call__(self, sentence) -> FilterMatch:
        ws = words(sentence)

        # Need at least three tokens: there + be + pivot.
        for i in range(len(ws) - 2):
            there, be, pivot = ws[i], ws[i + 1], ws[i + 2]

            # Slot 0: the expletive. The deprel ``expl`` nails it; otherwise we
            # accept a clause-initial "there" with a plausible tag, which keeps
            # the locative "there" (usually deprel advmod, sentence-final) out.
            if there.text.lower() != "there":
                continue
            is_expletive = there.deprel == "expl"
            clause_initial = i == 0 and there.upos in {"PRON", "DET", "ADV"}
            if not (is_expletive or clause_initial):
                continue

            # Slot 1: a form of "be". Lemma covers is/are/was/were/'s. The verb
            # agrees with the pivot, not with "there" — but we don't enforce that
            # here, it's exactly the contrast the eval items probe.
            if be.lemma != "be":
                continue

            # Slot 2: the pivot head. A numeral or determiner can open it
            # ("there are three ...", "there is no ..."), so we take any nominal
            # category rather than a plain NOUN.
            if pivot.upos not in _PIVOT_UPOS:
                continue

            note = f"there+{be.text}+{pivot.text} (existential pivot)"
            return FilterMatch.hit(i, i + 3, note=note)

        return FilterMatch.miss(note="no expletive-there + BE + pivot frame")


# Module-level instance so the package init can register it directly.
existential_there = ExistentialThereFilter()
