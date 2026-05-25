"""Sanity checks on the package's canonical constants and filter registry."""

from __future__ import annotations

import drc
from drc.data.filters import FILTERS, get_filter


def test_constructions_are_the_expected_codes():
    assert drc.CONSTRUCTIONS == (
        "aann",
        "comparative_correlative",
        "tough_movement",
        "resultative",
        "existential_there",
        "it_cleft",
        "negative_inversion",
        "subject_to_object_raising",
    )


def test_dose_levels():
    assert drc.DOSE_LEVELS == (0, 4, 16, 64, "all")


def test_seeds():
    assert drc.SEEDS == (42, 43, 44)


def test_filter_registry_keys_match_constructions():
    assert set(FILTERS) == set(drc.CONSTRUCTIONS)


def test_get_filter_returns_filter_with_matching_code():
    for code in drc.CONSTRUCTIONS:
        filt = get_filter(code)
        assert filt.code == code


def test_get_filter_rejects_unknown_code():
    try:
        get_filter("not_a_construction")
    except KeyError as exc:
        assert "not_a_construction" in str(exc)
    else:
        raise AssertionError("expected KeyError for an unknown construction code")
