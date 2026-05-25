"""Construction filters, keyed by their canonical short code.

Each filter is a small callable that takes a parsed Stanza sentence and returns
a ``FilterMatch`` (see ``base.py``). The corpus-generation and QA-audit code
look filters up by construction code, so we expose a single registry here rather
than importing the modules one by one everywhere.

The registry is validated against ``drc.CONSTRUCTIONS`` at import time: if a code
ever drifts, you find out immediately instead of silently filtering the wrong
construction.
"""

from __future__ import annotations

from ... import CONSTRUCTIONS
from .aann import aann
from .base import ConstructionFilter, FilterMatch
from .comparative_correlative import comparative_correlative
from .resultative import resultative
from .tough_movement import tough_movement

# Code -> filter instance. Keys must match the canonical short codes exactly.
FILTERS: dict[str, ConstructionFilter] = {
    aann.code: aann,
    comparative_correlative.code: comparative_correlative,
    tough_movement.code: tough_movement,
    resultative.code: resultative,
}

# Fail loud if the registry and the canonical code list ever diverge.
assert set(FILTERS) == set(CONSTRUCTIONS), (
    f"filter registry {sorted(FILTERS)} != CONSTRUCTIONS {sorted(CONSTRUCTIONS)}"
)


def get_filter(code: str) -> ConstructionFilter:
    """Return the filter registered under ``code``.

    Raises ``KeyError`` with the valid options listed, which beats a bare
    lookup error when someone mistypes a construction name on the CLI.
    """
    try:
        return FILTERS[code]
    except KeyError:
        raise KeyError(
            f"unknown construction code {code!r}; expected one of {sorted(FILTERS)}"
        ) from None


__all__ = ["FILTERS", "get_filter", "FilterMatch", "ConstructionFilter"]
