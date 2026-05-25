"""The experiment design --- exactly which models get trained, and why.

We deliberately avoid a full factorial. The headline quantity is \\(E_0\\), the
indirect-evidence floor: how well a model handles a construction it never saw
directly. Measuring \\(E_0\\) needs only the *zero-dose* model per construction
plus a ceiling reference --- it does not need the whole dose ladder. So the
design is tiered:

* **Breadth.** Every construction is trained at dose 0, giving \\(E_0\\) across a
  wide set of constructions. This is the sample for the indirect-evidence index
  and the (exploratory) learnability-prediction analysis.
* **Depth.** A representative *core* subset also gets the intermediate doses, so
  we can fit the full Hill curve and read off the threshold \\(E_{50}\\) and slope.
* **Shared ceiling.** One full-corpus model per seed is the \\(E_{\\max}\\)
  reference for *every* construction. At dose "all" only the target
  construction's count is left untouched, so that corpus is just the unfiltered
  corpus --- identical across constructions. Training it once instead of once per
  construction is the single biggest compute saving here.

The cell list lives in the config's ``design`` block so the compute envelope is
explicit and tunable without code changes. If that block is absent we fall back
to the legacy full factorial over the package constants, which keeps older
configs working.
"""

from __future__ import annotations

from typing import Any

# Pseudo-construction code for the shared full-corpus (ceiling) model. It trains
# on the unfiltered corpus and is evaluated on every construction's test set to
# supply each one's E_max point.
FULL_CORPUS_CODE = "full"


def design_block(config: dict[str, Any]) -> dict[str, Any]:
    """Return the design spec, falling back to a legacy full factorial."""
    block = config.get("design")
    if block:
        return block
    from drc import CONSTRUCTIONS, DOSE_LEVELS, SEEDS

    intermediate = [d for d in DOSE_LEVELS if d != "all"]
    return {
        "core_constructions": list(CONSTRUCTIONS),
        "breadth_constructions": [],
        "core_doses": intermediate,
        "breadth_doses": intermediate,
        "seeds": list(SEEDS),
        "shared_full_corpus": False,  # legacy: each construction trains its own "all"
        "_legacy": True,
    }


def all_constructions(config: dict[str, Any]) -> list[str]:
    """Every construction with evaluation items (core first, then breadth)."""
    d = design_block(config)
    return list(d["core_constructions"]) + list(d.get("breadth_constructions", []))


def core_constructions(config: dict[str, Any]) -> list[str]:
    return list(design_block(config)["core_constructions"])


def is_core(config: dict[str, Any], construction: str) -> bool:
    return construction in design_block(config)["core_constructions"]


def seeds(config: dict[str, Any]) -> list[int]:
    return list(design_block(config)["seeds"])


def shares_full_corpus(config: dict[str, Any]) -> bool:
    return bool(design_block(config).get("shared_full_corpus", False))


def curve_doses(config: dict[str, Any], construction: str) -> list[Any]:
    """The doses that make up one construction's dose-response series.

    Core constructions get the intermediate ladder; breadth constructions get
    only the doses listed under ``breadth_doses`` (dose 0 by default). The
    ceiling point is appended as ``"all"`` --- supplied by the shared full-corpus
    model when ``shared_full_corpus`` is set, otherwise by a per-construction
    "all" run (legacy).
    """
    d = design_block(config)
    base = d["core_doses"] if is_core(config, construction) else d.get("breadth_doses", [0])
    points = list(base)
    if "all" not in points:
        points.append("all")
    return points


def dose_cells(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """Unique ``(construction, dose)`` pairs that need a generated corpus.

    Excludes the shared ``"all"`` ceiling: when the full corpus is shared, only
    the ``full`` corpus is generated for the ceiling, not one per construction.
    """
    d = design_block(config)
    shared = shares_full_corpus(config)
    cells: list[tuple[str, Any]] = []
    if shared:
        cells.append((FULL_CORPUS_CODE, "all"))
    for c in all_constructions(config):
        base = d["core_doses"] if is_core(config, c) else d.get("breadth_doses", [0])
        for dose in base:
            cells.append((c, dose))
        if not shared and "all" not in base:
            cells.append((c, "all"))
    # De-dup while preserving order.
    seen: set[tuple[str, Any]] = set()
    return [x for x in cells if not (x in seen or seen.add(x))]


def run_cells(config: dict[str, Any], drop_dose: Any | None = None) -> list[tuple[str, Any, int]]:
    """Every ``(construction, dose, seed)`` model to train.

    ``drop_dose`` removes a dose from the *core* ladder for the single-GPU
    degradation path. The shared full-corpus model is emitted once per seed.
    """
    cells: list[tuple[str, Any, int]] = []
    ss = seeds(config)
    for construction, dose in dose_cells(config):
        if drop_dose is not None and dose == drop_dose and construction != FULL_CORPUS_CODE:
            continue
        for seed in ss:
            cells.append((construction, dose, seed))
    return cells


def emax_construction(config: dict[str, Any], construction: str) -> str:
    """Which trained model supplies the E_max point for this construction."""
    return FULL_CORPUS_CODE if shares_full_corpus(config) else construction
