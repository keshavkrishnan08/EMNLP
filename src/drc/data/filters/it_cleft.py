"""The it-cleft: "it BE XP that/who ...".

A cleft splits one clause into two, fronting a focused constituent behind an
expletive "it" and a copula, with the rest packaged into a relative-like clause:

    "it was John who left"
     ^^ ^^^ ^^^^ ^^^
     it  BE  focus REL

The "it" is non-referential — it doesn't point at anything — and the focused XP
("John") is what the sentence is really about. The trailing clause opens with a
relativizer ("that", "who", "which", "whom"). We match the surface frame:
sentence-initial "it", a form of BE, at least one token of focused material,
then a relativizer. We don't try to verify the gap inside the relative clause;
the it...BE...that frame is reliable enough on its own.

Reference: Lambrecht (2001), "A framework for the analysis of cleft
constructions," *Linguistics* 39(3).
"""

from __future__ import annotations

from .base import FilterMatch, words

# The words that can open the cleft's trailing clause. "that" is the default;
# "who"/"whom" front human foci, "which" non-human ones. We keep the set tight
# so an ordinary "it was nice that you came" extraposition doesn't leak in via a
# bare complementizer — those are caught only when a real focus XP precedes.
_RELATIVIZERS = frozenset({"that", "who", "whom", "which"})


class ItCleftFilter:
    """Detect the it + BE + focus-XP + relativizer cleft frame.

    We anchor on a sentence-initial "it", confirm the copula right after it, then
    scan forward for the relativizer. The focus XP is whatever sits between the
    copula and the relativizer; we just require it to be non-empty so the
    relativizer can't butt straight up against the copula.
    """

    code = "it_cleft"

    def __call__(self, sentence) -> FilterMatch:
        ws = words(sentence)

        # Need at minimum: it + be + focus + relativizer.
        if len(ws) < 4:
            return FilterMatch.miss(note="too short for it + BE + focus + REL")

        # Slot 0: the cleft "it", clause-initial and pronominal. Being first is
        # what distinguishes the cleft from a referential "it" mid-sentence.
        it = ws[0]
        if it.text.lower() != "it" or it.upos != "PRON":
            return FilterMatch.miss(note="no clause-initial pronominal 'it'")

        # Slot 1: a form of "be" immediately after "it". Lemma covers is/was/'s.
        be = ws[1]
        if be.lemma != "be":
            return FilterMatch.miss(note="'it' not followed by a form of BE")

        # Scan for the relativizer. Everything between the copula and it is the
        # focused XP; we require at least one such token so the focus isn't empty.
        for j in range(3, len(ws)):
            rel = ws[j]
            if rel.text.lower() in _RELATIVIZERS:
                focus_len = j - 2  # tokens strictly between BE and the relativizer
                if focus_len < 1:
                    continue
                note = (
                    f"it+{be.text}+<{focus_len}-tok focus>+{rel.text} (cleft)"
                )
                return FilterMatch.hit(0, j + 1, note=note)

        return FilterMatch.miss(note="no relativizer closing the cleft")


# Module-level instance so the package init can register it directly.
it_cleft = ItCleftFilter()
