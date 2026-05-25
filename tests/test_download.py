"""Unit tests for the BabyLM download helpers — pure, no network.

We can't hit the Hub in CI, but we can pin the parts that decide whether the
pulled data is shaped correctly: the domain-from-filename mapping, the
read/write contract (one ``domain<TAB>text`` line per record, internal tabs
flattened so the separator stays unambiguous), and the round-trip back through
the parser's reader.
"""

from __future__ import annotations

from drc.data.download import (
    DOMAIN_SEP,
    _domain_from_filename,
    _records_from_files,
    _truncate_to_words,
    _write_records,
)
from drc.data.parse import _iter_documents


def test_domain_from_filename_strips_train_suffixes():
    assert _domain_from_filename("open_subtitles.train.txt") == "open_subtitles"
    assert _domain_from_filename("childes.train") == "childes"
    assert _domain_from_filename("/tmp/snap/bnc_spoken.train.txt") == "bnc_spoken"


def test_records_carry_the_filename_domain(tmp_path):
    (tmp_path / "childes.train.txt").write_text("hi there\n\nsecond line\n", encoding="utf-8")
    (tmp_path / "switchboard.train.txt").write_text("uh huh\n", encoding="utf-8")
    records = _records_from_files(
        [tmp_path / "childes.train.txt", tmp_path / "switchboard.train.txt"]
    )
    # Blank lines are dropped; domains come from the filenames.
    assert records == [
        ("childes", "hi there"),
        ("childes", "second line"),
        ("switchboard", "uh huh"),
    ]


def test_write_then_read_round_trips_and_flattens_internal_tabs(tmp_path):
    """An internal tab in the source mustn't be confused with the domain marker."""
    records = [("switchboard", "B:\tYeah, right")]
    dest = tmp_path / "mini.txt"
    _write_records(records, dest)

    line = dest.read_text(encoding="utf-8").rstrip("\n")
    # Exactly one separator: the domain tag, then tab-free text.
    assert line.count(DOMAIN_SEP) == 1
    domain, _, text = line.partition(DOMAIN_SEP)
    assert domain == "switchboard"
    assert "\t" not in text

    # And the parser's reader recovers the domain.
    docs = list(_iter_documents(dest))
    assert docs == [("switchboard", "B: Yeah, right")]


def test_truncate_to_words_cuts_on_record_boundaries():
    records = [("d", "a b c"), ("d", "d e"), ("d", "f g h i")]
    kept = _truncate_to_words(records, max_words=5)
    # First record (3 words) fits, second (2) fits at 5; third would overflow.
    assert kept == [("d", "a b c"), ("d", "d e")]
