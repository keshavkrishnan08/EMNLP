"""Round-trip test for the Hill-curve fit: generate, fit, recover.

Skipped wholesale when scipy isn't around, since ``fit_one`` leans on
``scipy.optimize.curve_fit``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from drc.analysis.hill import fit_one, hill  # noqa: E402  (after importorskip guard)


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
