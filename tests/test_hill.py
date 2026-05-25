"""Round-trip test for the Hill-curve fit: generate, fit, recover.

Skipped wholesale when scipy isn't around, since ``fit_one`` leans on
``scipy.optimize.curve_fit``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

pd = pytest.importorskip("pandas")

from drc.analysis.hill import fit_all, fit_one, hill  # noqa: E402  (after importorskip guard)

# Tiered design: one core construction (full ladder) + one breadth (dose 0 only),
# both leaning on the shared "full" ceiling model for their E_max point.
_DESIGN_CONFIG = {
    "design": {
        "core_constructions": ["aann"],
        "breadth_constructions": ["it_cleft"],
        "core_doses": [0, 4, 16, 64],
        "breadth_doses": [0],
        "seeds": [42],
        "shared_full_corpus": True,
    }
}


def test_hill_curve_is_monotone_and_bounded():
    """Y rises from E0 toward Emax as dose grows."""
    D = np.array([0.0, 4.0, 16.0, 64.0, 256.0])
    Y = hill(D, E0=0.2, Emax=0.9, E50=16.0, n=2.0)
    assert Y[0] == pytest.approx(0.2, abs=1e-6)  # floor at D=0
    assert np.all(np.diff(Y) > 0)                # strictly increasing
    assert Y[-1] < 0.9 and Y[-1] > 0.85          # approaching the ceiling


def test_fit_recovers_known_parameters():
    """Fit clean data from a known curve and check we get the params back."""
    true = {"E0": 0.25, "Emax": 0.92, "E50": 16.0, "n": 2.0}
    D = np.array([0.0, 4.0, 16.0, 64.0, 256.0])
    Y = hill(D, **true)

    rng = np.random.default_rng(0)
    fit = fit_one(D, Y, rng)

    assert fit["converged"]
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-3)
    assert fit["E0"] == pytest.approx(true["E0"], abs=0.05)
    assert fit["Emax"] == pytest.approx(true["Emax"], abs=0.05)
    assert fit["E50"] == pytest.approx(true["E50"], rel=0.20)
    assert fit["n"] == pytest.approx(true["n"], rel=0.20)


def _write_eval_csv(path):
    """Synthetic eval_results.csv with a shared 'full' ceiling row.

    Core construction 'aann' has its dose ladder (0/4/16/64) plus the ceiling
    from the shared full model. Breadth 'it_cleft' has only dose 0 plus the same
    shared ceiling. The 'full' model is evaluated on each construction's pairs.
    """
    rows = [
        # aann's own model across the ladder (self-eval).
        ("aann", "0", "aann", 0.50),
        ("aann", "4", "aann", 0.62),
        ("aann", "16", "aann", 0.74),
        ("aann", "64", "aann", 0.85),
        # it_cleft's zero-dose model (self-eval).
        ("it_cleft", "0", "it_cleft", 0.66),
        # The shared full-corpus ceiling, evaluated on BOTH constructions.
        ("full", "all", "aann", 0.92),
        ("full", "all", "it_cleft", 0.88),
        # A decoy row that must be ignored (full evaluated cross-construction is
        # already the ceiling; here a stray dose value should not be picked up).
        ("aann", "16", "it_cleft", 0.01),
    ]
    df = pd.DataFrame(
        [
            {
                "model_construction": mc,
                "dose": d,
                "seed": 42,
                "eval_construction": ec,
                "n_pairs": 100,
                "n_correct": int(acc * 100),
                "accuracy": acc,
                "std_error": 0.02,
            }
            for mc, d, ec, acc in rows
        ]
    )
    df.to_csv(path, index=False)


def test_tiered_fit_uses_shared_ceiling_and_handles_breadth(tmp_path):
    """Core construction gets a fit using the 'full' ceiling as its 'all' point;
    breadth construction gets E0/Emax with converged=False and NaN E50/n."""
    eval_csv = tmp_path / "eval_results.csv"
    _write_eval_csv(eval_csv)

    fits = fit_all(eval_csv, sanity_path=None, config=_DESIGN_CONFIG, seed=0)

    aann = fits[fits["construction"] == "aann"].iloc[0]
    cleft = fits[fits["construction"] == "it_cleft"].iloc[0]

    # Core: fit converged, Emax pulled from the shared full ceiling (0.92).
    assert bool(aann["is_core"]) is True
    assert bool(aann["converged"]) is True
    assert aann["Emax"] == pytest.approx(0.92, abs=0.03)
    assert aann["E0"] == pytest.approx(0.50, abs=0.05)

    # Breadth: no fit. E0 = dose-0 accuracy, Emax = shared ceiling on it_cleft.
    assert bool(cleft["is_core"]) is False
    assert bool(cleft["converged"]) is False
    assert cleft["E0"] == pytest.approx(0.66, abs=1e-9)
    assert cleft["Emax"] == pytest.approx(0.88, abs=1e-9)
    assert np.isnan(cleft["E50"]) and np.isnan(cleft["n"])
