"""The comparative correlative: "the X-er ..., the Y-er ...".

Two clauses, each opening with "the" plus a comparative, paired so that one
covaries with the other:

    "the harder you try, the worse it gets"
     ^^^ ^^^^^^          ^^^ ^^^^^
     the COMP            the COMP

The "the" here isn't an ordinary article — it's a frozen degree marker, and the
construction is semi-idiomatic: the meaning ("as X increases, so does Y") isn't
fully predictable from the parts. We detect it structurally by finding two
"the + comparative" openings split by a comma, with the first clause sitting at
the start of the sentence. A regex fallback over the raw text picks up cases
where the parser mangles the unusual syntax, which boosts recall on the
construction's many irregular shapes ("the more the merrier").

References: Weissweiler et al. (2022), "The Better Your Syntax, the Better Your
Semantics?"; Goldberg (2003), "Constructions: a new theoretical approach to
language."
"""

from __future__ import annotations

import re

from .base import FilterMatch, feats_get, words

# Fallback pattern, applied to the lowercased sentence text. It mirrors the
# dependency check: clause-initial "the <comparative>", a comma, then a second
# "the <comparative>". We bound the gap so we don't match across unrelated
# sentences that merely happen to contain two "the"s.
_CC_REGEX = re.compile(
    r"^the\s+(\w+er|more|less)\s+.{1,60}?,\s*the\s+(\w+er|more|less)\b"
)

# Lexical comparatives that don't carry an -er suffix. Stanza usually tags these
# Degree=Cmp, but checking the surface form too keeps us robust when it doesn't.
_LEXICAL_CMP = {"more", "less"}


def _is_comparative(word) -> bool:
    """True if a word reads as a comparative adjective or adverb.

    We accept three signals: the morphological feature ``Degree=Cmp`` (the clean
    case), the bare lexical comparatives "more"/"less", or an ADJ/ADV that simply
    ends in "-er". The last is a cheap catch-all for parses that drop the feat.
    """
    text = word.text.lower()
    if text in _LEXICAL_CMP:
        return True
    if word.upos in {"ADJ", "ADV"}:
        if feats_get(word, "Degree") == "Cmp":
            return True
        # "-er" forms like "harder", "worse" is irregular so it won't catch, but
        # the feat check above usually does. The suffix test handles the rest.
        if text.endswith("er") and len(text) > 3:
            return True
    return False


class ComparativeCorrelativeFilter:
    """Detect the paired "the X-er, the Y-er" construction.

    Primary path is dependency/POS based: locate the two "the + comparative"
    openings around a comma. If that misses, we fall back to a regex so the
    construction's irregular variants still get counted.
    """

    code = "comparative_correlative"

    def __call__(self, sentence) -> FilterMatch:
        ws = words(sentence)

        # --- Primary, structure-aware detection -------------------------------
        # Find every "the" immediately followed by a comparative. The frozen
        # degree-marker "the" is what anchors each clause.
        the_cmp_idx: list[int] = []
        for i in range(len(ws) - 1):
            if ws[i].text.lower() == "the" and _is_comparative(ws[i + 1]):
                the_cmp_idx.append(i)

        if len(the_cmp_idx) >= 2:
            first, second = the_cmp_idx[0], the_cmp_idx[1]

            # The first clause should open the sentence. We allow a tiny offset
            # for a stray leading token but really expect index 0.
            clause_initial = first <= 1

            # A comma has to separate the two clauses — it's the orthographic
            # seam of the construction. Look for one strictly between the two
            # "the"s.
            comma_between = any(
                ws[j].text == "," for j in range(first + 1, second)
            )

            if clause_initial and comma_between:
                note = (
                    f"the+{ws[first + 1].text} ... , the+{ws[second + 1].text}"
                )
                return FilterMatch.hit(first, second + 2, note=note)

        # --- Regex fallback ---------------------------------------------------
        # The construction's syntax confuses parsers often enough that a surface
        # pattern earns real recall. Only reached when the structural check
        # didn't already fire.
        text = sentence.text.lower()
        if _CC_REGEX.search(text):
            return FilterMatch.hit(0, len(ws), note="regex fallback: the+CMP, the+CMP")

        return FilterMatch.miss(note="no paired the+comparative clauses")


comparative_correlative = ComparativeCorrelativeFilter()
