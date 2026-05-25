"""Subject-to-object raising (ECM): "I believe him to be honest".

A believe-class verb takes an accusative NP plus a to-infinitive, where the NP is
the *subject* of the embedded clause even though it surfaces as the matrix
object:

    "I believe him to be honest"
       ^^^^^^^ ^^^ ^^ ^^^^^^^^^
       V       NP  to embedded VP

"Him" gets accusative case from "believe" (hence the pronoun form), yet it's
understood as the one who is honest. This is also called exceptional case-marking
(ECM). The reliable surface frame is a verb from the raising/ECM class, then an
object NP, then "to", then a verb. We anchor on the verb's lemma and walk the
slots, allowing the object NP to be one or two tokens (a bare pronoun or a
determiner + noun).

Reference: Postal (1974), *On Raising: One Rule of English Grammar and Its
Theoretical Implications*.
"""

from __future__ import annotations

from .base import FilterMatch, words

# Verbs that take an ECM / subject-to-object-raising complement. A closed-ish
# class: believe-type epistemics plus a few declaratives ("declare", "report")
# that license the NP + to-infinitive frame. Lemmas, so inflection collapses.
_RAISING_VERBS = frozenset(
    {
        "believe",
        "consider",
        "expect",
        "find",
        "know",
        "declare",
        "prove",
        "suppose",
        "assume",
        "report",
    }
)

# Categories that can head the raised object NP. A pronoun ("him"), proper noun
# ("John"), common noun ("the dog" -> "dog"), or determiner ("it") all qualify.
_NP_UPOS = frozenset({"PRON", "PROPN", "NOUN", "DET"})


class SubjectToObjectRaisingFilter:
    """Detect a raising/ECM verb + object NP + "to" + embedded verb.

    Anchor on the matrix verb's lemma, then look ahead for the "to" + VERB tail,
    requiring an NP head somewhere in between. We allow a one- or two-token NP so
    both "believe him to..." and "consider the plan to..." match, but cap the gap
    so we don't span unrelated clauses.
    """

    code = "subject_to_object_raising"

    def __call__(self, sentence) -> FilterMatch:
        ws = words(sentence)

        # Need verb + NP + to + verb: at least four tokens from the verb on.
        for i in range(len(ws) - 3):
            verb = ws[i]

            # Slot 0: the matrix verb, identified by its lemma. The lexicon is
            # what separates ECM from ordinary "I want him to leave" (control) or
            # "I saw him leave" (bare-infinitive perception verbs).
            if verb.lemma not in _RAISING_VERBS or verb.upos not in {"VERB", "AUX"}:
                continue

            # Slot 1: the raised object NP head. It must be a nominal sitting
            # right after the verb.
            np = ws[i + 1]
            if np.upos not in _NP_UPOS:
                continue

            # The NP may be one token ("him") or two ("the dog"); locate the "to"
            # in the next slot or the one after it, then confirm a verb follows.
            for to_idx in (i + 2, i + 3):
                if to_idx + 1 >= len(ws):
                    break
                to_tok, emb = ws[to_idx], ws[to_idx + 1]
                if to_tok.lemma != "to" or to_tok.text.lower() != "to":
                    continue
                # The embedded predicate. "to be honest" -> AUX "be"; "to leave"
                # -> VERB. Accept either so copular complements count.
                if emb.upos not in {"VERB", "AUX"}:
                    continue
                note = (
                    f"{verb.text}+{np.text}+to+{emb.text} (ECM / raising-to-object)"
                )
                return FilterMatch.hit(i, to_idx + 2, note=note)

        return FilterMatch.miss(note="no raising-verb + NP + to + VERB frame")


# Module-level instance so the package init can register it directly.
subject_to_object_raising = SubjectToObjectRaisingFilter()
