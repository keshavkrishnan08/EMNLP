"""The AANN construction: Article + Adjective + Numeral + plural Noun.

English usually blocks an indefinite article in front of a plural noun
(*"a days"). But a narrow, productive pattern licenses exactly that, as long as
an adjective and a numeral sit in between:

    "a beautiful five days in Texas"
     ^   ^         ^    ^
     a   ADJ      NUM   NOUN.Plur

The semantics treat the plural as a measured quantity ("five days" behaves like
a single span). That's why the article can scope over the whole NP even though
its surface neighbor is plural. We match the four-token core and confirm the
article actually depends on the plural noun head, which rules out coincidental
sequences that span a clause boundary.

Reference: Mahowald (2023), "A Discerning Several Thousand Judgments: GPT-3
Rates the Article + Adjective + Numeral + Noun Construction."
"""

from __future__ import annotations

from .base import FilterMatch, feats_get, words


class AANNFilter:
    """Detect the a/an + ADJ + NUM + plural-NOUN sequence.

    The construction is rare but fully productive, so we lean on POS tags and a
    single dependency check rather than a word list. Numerals come in both word
    ("five") and digit ("5") form; Stanza tags both as ``NUM``, so we accept
    either — "a 5 minute walk" sits at the edge of the pattern but we keep it
    whenever the NUM slot is genuinely filled.
    """

    code = "aann"

    def __call__(self, sentence) -> FilterMatch:
        ws = words(sentence)

        # Slide a 4-token window across the sentence. The construction can start
        # anywhere (subject, object, after a preposition), so we don't anchor it.
        for i in range(len(ws) - 3):
            t0, t1, t2, t3 = ws[i], ws[i + 1], ws[i + 2], ws[i + 3]

            # Slot 0: the indefinite article, tagged DET. "the"/"some" don't
            # trigger the construction's signature article-over-plural clash.
            if t0.text.lower() not in {"a", "an"} or t0.upos != "DET":
                continue

            # Slot 1: a prenominal adjective. Without it the article can't reach
            # over the plural — *"a five days" is out.
            if t1.upos != "ADJ":
                continue

            # Slot 2: the numeral. This is the measure that lets the plural read
            # as one quantity.
            if t2.upos != "NUM":
                continue

            # Slot 3: the head noun, and it must be plural. That plurality is the
            # whole point — a singular noun here is just an ordinary NP.
            if t3.upos != "NOUN" or feats_get(t3, "Number") != "Plur":
                continue

            # Confirm the article really attaches to this noun. Stanza numbers
            # words from 1, so ``t0.head`` should equal ``t3.id``. This guards
            # against a window that straddles two NPs by accident. Post-modifiers
            # ("in Texas") are fine: they hang off the noun and don't touch this.
            if t0.head != t3.id:
                continue

            note = f"a/an+ADJ+NUM+NOUN.Plur: '{t0.text} {t1.text} {t2.text} {t3.text}'"
            return FilterMatch.hit(i, i + 4, note=note)

        return FilterMatch.miss(note="no a/an+ADJ+NUM+plural-noun span")


# Module-level instance so the package init can register it directly.
aann = AANNFilter()
