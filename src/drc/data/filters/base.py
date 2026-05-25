"""Shared scaffolding for construction filters.

Every construction filter takes a parsed sentence and decides whether the
sentence contains the target construction. We keep a common interface so the
corpus-generation and QA-audit code can treat all four filters identically.

A parsed sentence is a Stanza ``Sentence`` object. We never pass around raw
strings here: matching grammatical constructions reliably needs POS tags,
lemmas, and dependency arcs, and re-parsing inside a filter would be wasteful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FilterMatch:
    """One construction hit inside a sentence.

    ``span`` is a half-open token index range ``[start, end)`` over the
    sentence's words, useful for auditing and for highlighting matches when a
    human (or an LLM judge) checks the filter's precision.
    """

    matched: bool
    span: tuple[int, int] | None = None
    # Free-form notes that explain *why* the filter fired (or didn't). Handy
    # when debugging false positives during QA.
    note: str = ""

    @classmethod
    def miss(cls, note: str = "") -> FilterMatch:
        return cls(matched=False, span=None, note=note)

    @classmethod
    def hit(cls, start: int, end: int, note: str = "") -> FilterMatch:
        return cls(matched=True, span=(start, end), note=note)

    def __bool__(self) -> bool:  # lets callers write `if filter(sent): ...`
        return self.matched


@runtime_checkable
class ConstructionFilter(Protocol):
    """The contract every construction filter satisfies.

    Implementations are small callables (a function or a class with
    ``__call__``) so they're cheap to import and easy to unit-test in
    isolation. ``code`` must be one of ``drc.CONSTRUCTIONS``.
    """

    code: str

    def __call__(self, sentence) -> FilterMatch:  # sentence: stanza.models...Sentence
        ...


@dataclass
class FilterReport:
    """Precision/recall summary produced by the QA audit."""

    code: str
    n_positive: int
    precision: float
    recall: float
    n_audited: int
    notes: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """PRD acceptance threshold: precision >= 0.90, recall >= 0.85."""
        return self.precision >= 0.90 and self.recall >= 0.85


def words(sentence):
    """Return the list of Stanza ``Word`` objects for a sentence.

    Stanza nests words under tokens (a token can expand into multiple words,
    e.g. contractions). For construction matching we always want the word
    layer, so this small helper keeps the filters readable.
    """
    return list(sentence.words)


def feats_get(word, key: str) -> str | None:
    """Pull a single morphological feature (e.g. ``Number``) off a Stanza word.

    Stanza stores feats as a ``|``-joined string like ``"Number=Plur|Person=3"``.
    Returns ``None`` when the feature is absent.
    """
    if not word.feats:
        return None
    for chunk in word.feats.split("|"):
        name, _, value = chunk.partition("=")
        if name == key:
            return value
    return None
