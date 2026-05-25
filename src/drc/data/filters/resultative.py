"""The resultative: V + NP + AP, where the AP names the result state.

An action verb takes an object and a result-denoting adjective phrase that the
object ends up in:

    "he hammered the metal flat"
        ^^^^^^^^ ^^^^^^^^^ ^^^^
        VERB     obj       result AP

The metal becomes flat by way of the hammering — the adjective isn't an argument
of the verb on its own, it's the construction supplying a result slot. In a
dependency parse that shows up cleanly: the verb governs both an ``obj`` and an
``xcomp`` that's an adjective. We test exactly that, which avoids the noise of
trying to reason about result semantics from the surface string.

Reference: Goldberg & Jackendoff (2004), "The English Resultative as a Family
of Constructions."
"""

from __future__ import annotations

from .base import FilterMatch, words


class ResultativeFilter:
    """Detect a verb governing both an object and an adjectival xcomp.

    Iterate over verbs, gather each verb's dependents, and require the two arcs
    that define the construction: an ``obj`` (the affected NP) and an ``xcomp``
    that's an ``ADJ`` (the result state). Both hanging off the same verb is the
    structural signature — a predicative adjective elsewhere won't qualify.
    """

    code = "resultative"

    def __call__(self, sentence) -> FilterMatch:
        ws = words(sentence)

        for verb in ws:
            if verb.upos != "VERB":
                continue

            # Children of this verb: words whose head points back to it. Stanza
            # ids are 1-based, so the comparison is on raw ids.
            children = [w for w in ws if w.head == verb.id]

            obj = next((w for w in children if w.deprel == "obj"), None)
            # The result AP attaches as an open clausal complement (xcomp) and is
            # an adjective — that pairing is what marks the resultative.
            result = next(
                (w for w in children if w.deprel == "xcomp" and w.upos == "ADJ"),
                None,
            )

            if obj is not None and result is not None:
                # Span the verb through the result adjective so the match covers
                # the whole V + NP + AP frame. Guard the ordering since the
                # object and result can sit on either side of the verb.
                ids = [verb.id, obj.id, result.id]
                start = min(ids) - 1  # back to 0-based for the token range
                end = max(ids)        # half-open: last index + 1
                note = (
                    f"{verb.text} +obj '{obj.text}' +xcomp.ADJ '{result.text}'"
                )
                return FilterMatch.hit(start, end, note=note)

        return FilterMatch.miss(note="no verb with both obj and adjectival xcomp")


resultative = ResultativeFilter()
