"""Tests for the cross-construction transfer module.

Build a tiny full eval table in tmp_path, run the slope fitter and the matrix
builder, and check shapes/columns plus a known-monotone case yielding a
positive on-target slope. Skipped when numpy/pandas aren't around.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np  # noqa: E402  (after importorskip guard)
import pandas as pd  # noqa: E402

from drc import CONSTRUCTIONS  # noqa: E402
from drc.analysis.transfer import (  # noqa: E402
    _tidy_matrix,
    build_transfer,
    fit_slope,
)


def test_fit_slope_positive_on_monotone():
    """Accuracy rising with dose gives a clear positive slope."""
    dose = np.array([0.0, 4.0, 16.0, 64.0, 256.0])
    # Accuracy increases monotonically with log dose.
    acc = 0.5 + 0.1 * np.log10(dose + 1.0)
    slope, stderr = fit_slope(dose, acc)
    assert slope == pytest.approx(0.1, abs=1e-6)
    assert stderr == pytest.approx(0.0, abs=1e-6)


def test_fit_slope_degenerate_returns_nan():
    slope, stderr = fit_slope(np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.6, 0.7]))
    assert np.isnan(slope) and np.isnan(stderr)


def _synthetic_eval_df():
    """Full cross-eval table: on-target rises, off-target flat or falling."""
    doses = [0, 4, 16, 64, "all"]
    dose_value_map = {0: 0.0, 4: 4.0, 16: 16.0, 64: 64.0, "all": 256.0}
    rng = np.random.default_rng(0)
    rows = []
    for train in CONSTRUCTIONS:
        for ev in CONSTRUCTIONS:
            for dose in doses:
                dv = dose_value_map[dose]
                lx = np.log10(dv + 1.0)
                if train == ev:
                    base = 0.5 + 0.08 * lx  # on-target climbs
                else:
                    base = 0.5 - 0.01 * lx  # off-target drifts slightly down
                for seed in (42, 43, 44):
                    rows.append(
                        {
                            "model_construction": train,
                            "eval_construction": ev,
                            "dose": dose,
                            "dose_value": dv,
                            "seed": seed,
                            "accuracy": float(base + rng.normal(0, 1e-4)),
                        }
                    )
    return pd.DataFrame(rows)


def test_build_transfer_shape_and_diagonal():
    eval_df = _synthetic_eval_df()
    slopes, stderrs = build_transfer(eval_df)

    assert slopes.shape == (len(CONSTRUCTIONS), len(CONSTRUCTIONS))
    assert stderrs.shape == slopes.shape
    assert list(slopes.index) == list(CONSTRUCTIONS)
    assert list(slopes.columns) == list(CONSTRUCTIONS)

    # On-target (diagonal) slopes are positive; off-target are negative here.
    for c in CONSTRUCTIONS:
        assert slopes.loc[c, c] > 0
    for train in CONSTRUCTIONS:
        for ev in CONSTRUCTIONS:
            if train != ev:
                assert slopes.loc[train, ev] < 0


def test_tidy_matrix_columns():
    eval_df = _synthetic_eval_df()
    slopes, stderrs = build_transfer(eval_df)
    tidy = _tidy_matrix(slopes, stderrs)

    assert len(tidy) == len(CONSTRUCTIONS) ** 2
    for col in ("model_construction", "eval_construction", "slope", "stderr", "is_diagonal"):
        assert col in tidy.columns
    assert tidy["is_diagonal"].sum() == len(CONSTRUCTIONS)
