"""Pit the Hill curve against simpler shapes and let AIC/BIC referee.

A good Hill fit on its own proves little. Maybe the data is just a straight line
in log-dose, or a single step, or pure noise around a mean. So we fit each
candidate to the same (construction, seed) cells and rank them by information
criterion. The candidates:

* **hill** — the four-parameter curve from ``hill.py``, the hypothesis of interest.
* **power** — ``a*(D+1)^b + c``, a scale-free rise with no saturation.
* **step** — flat low, jump, flat high. The crudest "threshold" model; if a real
  phase transition exists, step should be competitive with Hill.
* **log-linear** — ``a + b*log(D+1)``, the boring "more data helps, smoothly" story.
* **null** — the mean. The floor every other model has to beat to earn its parameters.

We score with Gaussian log-likelihood (residual sum of squares under a fitted
noise variance), then AIC and BIC with the right parameter counts. Lower wins.
The point isn't to crown Hill — it's to give the decision rule honest evidence,
including the chance that something simpler explains the curves just as well.

CLI::

    python -m drc.analysis.model_comparison --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drc import CONSTRUCTIONS
from drc.analysis.hill import hill, resolve_dose_values
from drc.data.download import load_config, resolve_path

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


# --- candidate shapes ------------------------------------------------------
# Each returns (predicted Y, number of free parameters, converged flag). They
# share a tiny least-squares wrapper so the comparison is apples-to-apples.

def _power(D, a, b, c):
    import numpy as np
    return a * np.power(np.asarray(D, dtype=float) + 1.0, b) + c


def _log_linear(D, a, b):
    import numpy as np
    return a + b * np.log(np.asarray(D, dtype=float) + 1.0)


def _step(D, lo, hi, thresh):
    """Two flat levels with a switch at ``thresh``. Not differentiable, so we
    fit it by a coarse threshold search rather than gradient descent."""
    import numpy as np
    D = np.asarray(D, dtype=float)
    return np.where(D >= thresh, hi, lo)


def _fit_curve(func: Callable, D, Y, p0, bounds=None) -> np.ndarray | None:
    from scipy.optimize import curve_fit
    try:
        kw: dict[str, Any] = {"p0": p0, "maxfev": 20000}
        if bounds is not None:
            kw["bounds"] = bounds
            kw["method"] = "trf"
        popt, _ = curve_fit(func, D, Y, **kw)
        return popt
    except Exception as exc:  # noqa: BLE001
        logger.debug("Fit failed for %s: %s", getattr(func, "__name__", func), exc)
        return None


def _gaussian_loglik(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    """Log-likelihood of residuals under a fitted-variance Gaussian.

    Standard nonlinear-regression trick: the MLE noise variance is RSS/n, which
    collapses the log-likelihood to a clean function of the residual sum of
    squares. Lets us compare any two shapes on equal footing.
    """
    import numpy as np

    n = Y.size
    rss = float(np.sum((Y - Y_hat) ** 2))
    if rss <= 0:  # perfect fit — nudge so the log stays finite
        rss = 1e-12
    sigma2 = rss / n
    return -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)


def _aic_bic(loglik: float, k: int, n: int) -> tuple[float, float]:
    import numpy as np
    aic = 2.0 * k - 2.0 * loglik
    bic = k * np.log(n) - 2.0 * loglik
    return aic, bic


def fit_models(D: np.ndarray, Y: np.ndarray) -> list[dict[str, Any]]:
    """Fit all five candidates to one cell and score each.

    Free-parameter counts include the noise variance, since that's estimated
    too — keeping AIC/BIC honest across models with different shapes.
    """
    import numpy as np

    n = Y.size
    out: list[dict[str, Any]] = []

    def record(model: str, Y_hat, k_shape: int, converged: bool) -> None:
        if not converged or Y_hat is None:
            out.append({"model": model, "loglik": np.nan, "aic": np.nan,
                        "bic": np.nan, "n_params": k_shape + 1, "converged": False})
            return
        ll = _gaussian_loglik(Y, np.asarray(Y_hat, dtype=float))
        k = k_shape + 1  # +1 for the estimated variance
        aic, bic = _aic_bic(ll, k, n)
        out.append({"model": model, "loglik": ll, "aic": aic, "bic": bic,
                    "n_params": k, "converged": True})

    # hill (4 shape params)
    p = _fit_curve(
        hill, D, Y,
        p0=[float(Y.min()), float(Y.max()), float(np.median(D[D > 0]) if np.any(D > 0) else 1.0), 1.0],
        bounds=([0, 0, 1e-6, 1e-3], [1, 1, np.inf, 50.0]),
    )
    record("hill", hill(D, *p) if p is not None else None, 4, p is not None)

    # power law (3 shape params)
    p = _fit_curve(_power, D, Y, p0=[0.1, 0.5, float(Y.min())])
    record("power", _power(D, *p) if p is not None else None, 3, p is not None)

    # log-linear (2 shape params)
    p = _fit_curve(_log_linear, D, Y, p0=[float(Y.min()), 0.05])
    record("log_linear", _log_linear(D, *p) if p is not None else None, 2, p is not None)

    # step (2 levels + 1 threshold). Brute-force the threshold across the gaps
    # between sorted doses, fitting the two levels as plain means each side.
    best = None
    order = np.argsort(D)
    Ds, Ys = D[order], Y[order]
    candidates = (Ds[:-1] + Ds[1:]) / 2.0 if Ds.size > 1 else np.array([Ds.mean()])
    for t in candidates:
        mask = Ds >= t
        if mask.all() or (~mask).all():
            continue
        lo, hi = Ys[~mask].mean(), Ys[mask].mean()
        rss = float(np.sum((Ys - np.where(mask, hi, lo)) ** 2))
        if best is None or rss < best[0]:
            best = (rss, lo, hi, t)
    if best is not None:
        _, lo, hi, t = best
        record("step", _step(D, lo, hi, t), 3, True)
    else:
        record("step", None, 3, False)

    # null: mean only (1 shape param)
    record("null", np.full_like(Y, Y.mean()), 1, True)

    return out


def compare_all(eval_csv: Path, sanity_path: Path | None) -> pd.DataFrame:
    """Run the comparison over every (construction, seed) cell."""
    import pandas as pd

    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Eval results not found at {eval_csv}. Run the eval stage first."
        )

    df = pd.read_csv(eval_csv)
    df = df[df["model_construction"] == df["eval_construction"]].copy()
    df = resolve_dose_values(df, sanity_path)

    rows: list[dict[str, Any]] = []
    for cons in CONSTRUCTIONS:
        for s in sorted(df.loc[df["model_construction"] == cons, "seed"].unique()):
            cell = df[(df["model_construction"] == cons) & (df["seed"] == s)]
            cell = cell.sort_values("dose_value")
            if len(cell) < 4:
                logger.warning("Skipping %s seed %s: %d dose points.", cons, s, len(cell))
                continue
            D = cell["dose_value"].to_numpy(dtype=float)
            Y = cell["accuracy"].to_numpy(dtype=float)
            for rec in fit_models(D, Y):
                rows.append({"construction": cons, "seed": int(s), **rec})

    if not rows:
        raise RuntimeError("No cells had enough dose points for model comparison.")
    return pd.DataFrame(rows)


def run(config_path: Path) -> Path:
    """Run the comparison and write ``results/model_comparison.csv``."""
    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    eval_csv = results_dir / "eval_results.csv"
    sanity_path = results_dir / "dose_corpora_sanity.json"

    table = compare_all(eval_csv, sanity_path)
    out_path = results_dir / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logger.info("Wrote model comparison (%d rows) to %s", len(table), out_path)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
