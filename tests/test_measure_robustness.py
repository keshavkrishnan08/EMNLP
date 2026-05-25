"""Tests for the second-measure (mean-LP) robustness check.

Two things get pinned down here. First, the analysis itself: from a synthetic
``eval_per_item.csv`` carrying both ``correct`` and ``correct_meanlp`` flags (and
a shared 'full' ceiling), the per-cell accuracies, the agreement summary, and the
output CSV columns all come out right. Second, run_eval's per-item row helper now
emits the three new mean-LP columns — exercised with mocked scores, no torch.

Skipped wholesale when pandas/numpy/scipy aren't installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pd = pytest.importorskip("pandas")

from drc.analysis.measure_robustness import (  # noqa: E402
    agreement_summary,
    compute,
)

# One core construction with a full dose ladder, sharing the 'full' ceiling. A
# second core construction lets the E0-ordering rank correlation have something
# to rank.
DESIGN_CONFIG = {
    "design": {
        "core_constructions": ["aann", "tough_movement"],
        "breadth_constructions": [],
        "core_doses": [0, 4, 16, 64],
        "breadth_doses": [0],
        "seeds": [42, 43],
        "shared_full_corpus": True,
    }
}


def _row(mc, dose, seed, ec, item, correct, correct_meanlp):
    """One per-item record; the raw scores are placeholders we don't read."""
    return {
        "model_construction": mc,
        "dose": dose,
        "seed": seed,
        "eval_construction": ec,
        "item_id": item,
        "slor_good": 1.0,
        "slor_bad": 0.0,
        "correct": correct,
        "meanlp_good": 1.0,
        "meanlp_bad": 0.0,
        "correct_meanlp": correct_meanlp,
    }


def _write_per_item(path: Path):
    """Two items per cell, both measures, both seeds, plus the 'full' ceiling.

    Accuracies are engineered: aann climbs 0.5 -> 1.0 with dose, tough_movement
    sits higher at dose-0. SLOR and mean-LP are made to agree everywhere except
    one item, so the correlation is high but not a degenerate 1.0.
    """
    rows = []
    # aann self-eval ladder: dose-0 half right, higher doses all right.
    for seed in (42, 43):
        rows.append(_row("aann", "0", seed, "aann", "i1", 1, 1))
        rows.append(_row("aann", "0", seed, "aann", "i2", 0, 0))  # acc 0.5
        for dose in ("4", "16", "64"):
            rows.append(_row("aann", dose, seed, "aann", "i1", 1, 1))
            rows.append(_row("aann", dose, seed, "aann", "i2", 1, 1))  # acc 1.0
        # tough_movement dose-0 higher (0.75), one item where the measures split.
        rows.append(_row("tough_movement", "0", seed, "tough_movement", "i1", 1, 1))
        rows.append(_row("tough_movement", "0", seed, "tough_movement", "i2", 1, 0))
        for dose in ("4", "16", "64"):
            rows.append(_row("tough_movement", dose, seed, "tough_movement", "i1", 1, 1))
            rows.append(_row("tough_movement", dose, seed, "tough_movement", "i2", 1, 1))
        # Shared full ceiling, scored on each construction.
        rows.append(_row("full", "all", seed, "aann", "i1", 1, 1))
        rows.append(_row("full", "all", seed, "aann", "i2", 1, 1))
        rows.append(_row("full", "all", seed, "tough_movement", "i1", 1, 1))
        rows.append(_row("full", "all", seed, "tough_movement", "i2", 1, 1))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_per_cell_accuracies(tmp_path):
    per_item = tmp_path / "eval_per_item.csv"
    _write_per_item(per_item)

    cells = compute(per_item, DESIGN_CONFIG)

    assert set(cells.columns) == {"construction", "dose", "acc_slor", "acc_meanlp", "n"}
    # Five dose points per core construction (0,4,16,64,all), two constructions.
    assert len(cells) == 10

    by = cells.set_index(["construction", "dose"])
    # aann dose-0: 1 of 2 items correct under both measures, pooled over 2 seeds.
    assert by.loc[("aann", "0"), "acc_slor"] == pytest.approx(0.5)
    assert by.loc[("aann", "0"), "acc_meanlp"] == pytest.approx(0.5)
    assert by.loc[("aann", "0"), "n"] == 4  # 2 items x 2 seeds
    # aann ceiling comes from the 'full' rows, not an aann 'all' model.
    assert by.loc[("aann", "all"), "acc_slor"] == pytest.approx(1.0)
    # tough_movement dose-0: SLOR all right (1.0); mean-LP splits one item (0.75).
    assert by.loc[("tough_movement", "0"), "acc_slor"] == pytest.approx(1.0)
    assert by.loc[("tough_movement", "0"), "acc_meanlp"] == pytest.approx(0.75)


def test_agreement_summary(tmp_path):
    per_item = tmp_path / "eval_per_item.csv"
    _write_per_item(per_item)
    cells = compute(per_item, DESIGN_CONFIG)

    summary = agreement_summary(cells)

    assert set(summary.columns) == {
        "n_cells", "n_constructions_e0", "pearson", "spearman",
        "e0_rank_spearman", "mean_abs_diff",
    }
    assert summary.shape[0] == 1
    assert summary["n_cells"].iloc[0] == 10
    assert summary["n_constructions_e0"].iloc[0] == 2
    # The measures track each other closely; correlation should be strong.
    assert summary["pearson"].iloc[0] > 0.9
    # Mean absolute accuracy gap is small (only one cell differs, by 0.25).
    assert summary["mean_abs_diff"].iloc[0] == pytest.approx(0.025, abs=1e-9)


def test_writes_both_csvs_and_figure(tmp_path):
    """run() drops both CSVs and the figure given a config-shaped results dir."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import yaml

    from drc.analysis.measure_robustness import run

    results = tmp_path / "results"
    results.mkdir()
    _write_per_item(results / "eval_per_item.csv")

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "base.yaml"
    config = dict(DESIGN_CONFIG)
    config["paths"] = {"results": "results"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    out = run(config_path)
    assert out["cells"].exists()
    assert out["summary"].exists()
    assert out["figure"].exists()
    assert out["figure"].name == "fig10_measure_robustness.pdf"


def test_missing_per_item_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute(tmp_path / "nope.csv", DESIGN_CONFIG)


def test_run_eval_per_item_helper_emits_meanlp_columns():
    """The per-item row helper now records mean-LP good/bad and correct_meanlp.

    Torch-free: we feed mocked five-tuples and check the new columns and the
    second correctness rule (mean-LP good > bad). The older three-tuple form
    still works, leaving the mean-LP cells blank.
    """
    from drc.eval.run_eval import PER_ITEM_FIELDS, ModelRun, _per_item_rows

    run_info = ModelRun(construction="aann", dose="4", seed=42, path=Path("x"))
    scored = [
        ("i1", 2.0, 1.0, 0.8, 0.3),   # both measures: good wins
        ("i2", 0.5, 0.9, 0.4, 0.6),   # both measures: bad wins
        ("i3", 1.0, 0.5, 0.2, 0.7),   # SLOR good wins, mean-LP bad wins
    ]
    rows = _per_item_rows(run_info, "aann", scored)

    assert [tuple(r.keys()) for r in rows] == [PER_ITEM_FIELDS] * 3
    assert {"meanlp_good", "meanlp_bad", "correct_meanlp"} <= set(rows[0])
    assert rows[0]["correct"] == 1 and rows[0]["correct_meanlp"] == 1
    assert rows[1]["correct"] == 0 and rows[1]["correct_meanlp"] == 0
    # Measures can disagree per item — that's the whole point of recording both.
    assert rows[2]["correct"] == 1 and rows[2]["correct_meanlp"] == 0
    assert rows[0]["meanlp_good"] == pytest.approx(0.8)

    # Backward compatibility: a three-tuple leaves the mean-LP cells blank.
    legacy = _per_item_rows(run_info, "aann", [("i9", 2.0, 1.0)])
    assert legacy[0]["correct"] == 1
    assert legacy[0]["meanlp_good"] == ""
    assert legacy[0]["correct_meanlp"] == ""
