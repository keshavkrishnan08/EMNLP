"""Shared fixtures and lightweight Stanza stand-ins for the test suite.

The construction filters expect a Stanza ``Sentence`` (words with ``.text``,
``.upos``, ``.lemma``, ``.feats``, ``.id``, ``.head``, ``.deprel``). Stanza is a
heavy dependency we don't want in CI, so we fake just enough of that shape to
unit-test the pure matching logic. The filters never touch anything beyond those
attributes, so a couple of dataclasses do the job.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# Make `import drc` work without an install step. The package lives under src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass
class FakeWord:
    """A stand-in for a Stanza ``Word``.

    Only the attributes the filters read are modelled. ``id`` and ``head`` are
    1-based like Stanza's, where ``head == 0`` means the root.
    """

    text: str
    upos: str
    lemma: str
    feats: str | None
    id: int
    head: int
    deprel: str


@dataclass
class FakeSentence:
    """A stand-in for a Stanza ``Sentence``: a list of words plus surface text."""

    words: list[FakeWord]
    text: str


def make_sentence(rows, text: str | None = None) -> FakeSentence:
    """Build a FakeSentence from ``(text, upos, lemma, feats, deprel, head)`` rows.

    Word ids are assigned 1..N in order, matching Stanza's 1-based indexing, so
    ``head`` values in the rows can reference those ids directly. When ``text``
    isn't given we join the token texts with spaces — good enough for the
    text-based regex path in the comparative-correlative filter.
    """
    words = [
        FakeWord(
            text=tok,
            upos=upos,
            lemma=lemma,
            feats=feats,
            id=i,
            head=head,
            deprel=deprel,
        )
        for i, (tok, upos, lemma, feats, deprel, head) in enumerate(rows, start=1)
    ]
    if text is None:
        text = " ".join(w.text for w in words)
    return FakeSentence(words=words, text=text)


@pytest.fixture
def make_sentence_fixture():
    """Expose ``make_sentence`` to tests as a fixture (clean import path)."""
    return make_sentence
