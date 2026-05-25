"""Tests for the indirect-evidence index (E0 / Emax / direct_value).

We build a synthetic eval_results.csv that includes a shared
``model_construction == "full"`` ceiling row, then check the index reads E0 from
the zero-dose model, Emax from the 'full' row, and computes direct_value and the
leakage bound correctly. Skipped wholesale when pandas isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from drc.analysis.indirect_evidence import (  # noqa: E402
    compute_indirect_evidence,
    filter_recall,
)

# One core + one breadth construction, both sharing the 'full' ceiling.
DESIGN_CONFIG = {
    "design": {
        "core_constructions": ["aann"],
        "breadth_constructions": ["it_cleft"],
        "core_doses": [0, 4, 16, 64],
        "breadth_doses": [0],
        "seeds": [42, 43],
        "shared_full_corpus": True,
    }
}


def _write_eval_csv(path):
    rows = [
        # aann dose-0 over two seeds (E0 = mean = 0.50, with SE > 0).
        ("aann", "0", 42, "aann", 0.48),
        ("aann", "0", 43, "aann", 0.52),
        # aann ladder (not needed for the index, but realistic).
        ("aann", "64", 42, "aann", 0.80),
        # it_cleft dose-0 over two seeds (E0 = 0.60).
        ("it_cleft", "0", 42, "it_cleft", 0.60),
        ("it_cleft", "0", 43, "it_cleft", 0.60),
        # The shared full ceiling, evaluated on each construction, per seed.
        ("full", "all", 42, "aann", 0.90),
        ("full", "all", 43, "aann", 0.94),
        ("full", "all", 42, "it_cleft", 0.85),
        ("full", "all", 43, "it_cleft", 0.87),
    ]
    df = pd.DataFrame(
        [
            {
                "model_construction": mc,
                "dose": d,
                "seed": s,
                "eval_construction": ec,
                "n_pairs": 100,
                "n_correct": int(acc * 100),
                "accuracy": acc,
                "std_error": 0.02,
            }
            for mc, d, s, ec, acc in rows
        ]
    )
    df.to_csv(path, index=False)


def test_index_uses_full_row_for_emax(tmp_path):
    eval_csv = tmp_path / "eval_results.csv"
    _write_eval_csv(eval_csv)

    table = compute_indirect_evidence(
        eval_csv, DESIGN_CONFIG, audits_dir=None, default_recall=0.85
    )

    by = table.set_index("construction")

    # E0 = mean of dose-0 accuracies across seeds.
    assert by.loc["aann", "E0"] == pytest.approx(0.50, abs=1e-9)
    assert by.loc["it_cleft", "E0"] == pytest.approx(0.60, abs=1e-9)
    # E0_se > 0 for aann (0.48/0.52 differ), 0 for it_cleft (identical).
    assert by.loc["aann", "E0_se"] > 0
    assert by.loc["it_cleft", "E0_se"] == pytest.approx(0.0, abs=1e-12)

    # Emax comes from the shared 'full' ceiling rows (mean over seeds), NOT the
    # construction's own model.
    assert by.loc["aann", "Emax"] == pytest.approx((0.90 + 0.94) / 2, abs=1e-9)
    assert by.loc["it_cleft", "Emax"] == pytest.approx((0.85 + 0.87) / 2, abs=1e-9)

    # direct_value = Emax - E0.
    assert by.loc["aann", "direct_value"] == pytest.approx(0.92 - 0.50, abs=1e-9)
    assert by.loc["it_cleft", "direct_value"] == pytest.approx(0.86 - 0.60, abs=1e-9)

    # Leakage caveat present, flagged as a default-sourced upper bound.
    assert (table["filter_recall"] == 0.85).all()
    assert table["leakage_bound_note"].str.contains("UPPER BOUND").all()
    assert table["leakage_bound_note"].str.contains("default").all()


def test_missing_eval_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_indirect_evidence(
            tmp_path / "nope.csv", DESIGN_CONFIG, audits_dir=None
        )


def test_missing_ceiling_row_raises(tmp_path):
    """No 'full' ceiling row -> Emax can't be read, so we stop loudly."""
    eval_csv = tmp_path / "eval_results.csv"
    df = pd.DataFrame(
        [
            {
                "model_construction": "aann",
                "dose": "0",
                "seed": 42,
                "eval_construction": "aann",
                "n_pairs": 100,
                "n_correct": 50,
                "accuracy": 0.50,
                "std_error": 0.02,
            }
        ]
    )
    df.to_csv(eval_csv, index=False)
    with pytest.raises(RuntimeError):
        compute_indirect_evidence(eval_csv, DESIGN_CONFIG, audits_dir=None)


def test_filter_recall_scores_audit(tmp_path):
    """A labeled audit CSV is scored for recall; source flagged 'audit'."""
    audits = tmp_path / "qa_audits"
    audits.mkdir()
    # 1 true positive caught, 1 missed (fn) -> recall = 0.5.
    csv_path = audits / "aann_audit.csv"
    csv_path.write_text(
        "sentence_id,sentence_text,filter_label,human_label,notes\n"
        "1,hit caught,1,1,\n"
        "2,hit missed,0,1,\n"
        "3,clean,0,0,\n",
        encoding="utf-8",
    )
    recall, source = filter_recall(audits, "aann", default=0.85)
    assert source == "audit"
    assert recall == pytest.approx(0.5, abs=1e-9)

    # Missing audit -> default, flagged.
    recall2, source2 = filter_recall(audits, "it_cleft", default=0.85)
    assert source2 == "default"
    assert recall2 == pytest.approx(0.85, abs=1e-9)
