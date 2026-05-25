"""Fit the four-parameter Hill equation to each construction's dose-response.

The Hill equation is the workhorse curve in pharmacology, and it maps cleanly
onto our question. Swap "drug concentration" for "number of construction
exposures in pretraining" and "response" for "minimal-pair accuracy" and you get

    Y(D) = E0 + (Emax - E0) * D^n / (E50^n + D^n)

with four readable parameters. ``E0`` is the floor — what the model scores with
zero exposure (pure indirect evidence). ``Emax`` is the ceiling it saturates at.
``E50`` is the dose that gets you halfway up, an interpretable "how much data does
this construction need" number. And ``n``, the Hill coefficient, is the shape
knob: near 1 the curve is a gentle saturating rise, much larger than that and it
sharpens into a step — the signature we'd call a phase transition.

We fit one curve per (construction, seed) so seed-to-seed spread becomes a real
distribution, not a footnote. Confidence intervals come from a parametric
bootstrap: resample residuals under the fitted curve, refit, repeat. That's
slower than the curve_fit covariance but it doesn't lean on the asymptotic
normality assumption, which is shaky with only five dose points.

CLI::

    python -m drc.analysis.hill --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drc import CONSTRUCTIONS
from drc.data.download import load_config, resolve_path

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps numpy/pandas lazy
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

# Bootstrap resamples for the 95% CIs. A thousand is enough to stabilise the
# 2.5/97.5 percentiles without making the full sweep painfully slow.
N_BOOTSTRAP = 1000

# Fallback attested counts for dose="all" when the sanity JSON is absent. These
# are the corpus-wide totals recorded during dose generation; documented here so
# a missing file degrades gracefully instead of crashing. Update alongside any
# corpus regeneration.
ATTESTED_ALL_COUNTS: dict[str, int] = {
    "aann": 1200,
    "comparative_correlative": 340,
    "tough_movement": 890,
    "resultative": 2100,
}

# Parameter order used everywhere below, so the fit vector, bounds, and CSV
# columns can never drift out of sync.
PARAM_NAMES = ("E0", "Emax", "E50", "n")


def hill(D, E0, Emax, E50, n):
    """The four-parameter Hill curve. Vectorised over dose ``D``.

    Kept module-level (not a closure) so the bootstrap can pickle it and so the
    Julia mirror has an obvious counterpart to match.
    """
    import numpy as np

    D = np.asarray(D, dtype=float)
    # Clip the exponent base away from exactly zero; D=0 with n<1 would blow up.
    Dn = np.power(np.clip(D, 0.0, None), n)
    E50n = np.power(E50, n)
    return E0 + (Emax - E0) * Dn / (E50n + Dn)


def resolve_dose_values(
    df: pd.DataFrame, sanity_path: Path | None
) -> pd.DataFrame:
    """Replace the literal dose="all" with the construction's attested count.

    Every other dose is already numeric. ``"all"`` means "every instance we
    found in the corpus", and the actual number differs per construction, so we
    look it up — from ``results/dose_corpora_sanity.json`` when it's there, and
    from the documented fallback table when it isn't.
    """
    import numpy as np

    counts = dict(ATTESTED_ALL_COUNTS)
    if sanity_path is not None and sanity_path.exists():
        with open(sanity_path, encoding="utf-8") as fh:
            sanity = json.load(fh)
        # The sanity file maps construction -> {..., "attested_all": int} or a
        # bare int. Accept both shapes; warn and keep the fallback otherwise.
        for cons, payload in sanity.items():
            if isinstance(payload, dict):
                val = payload.get("attested_all") or payload.get("n_all")
            else:
                val = payload
            if val is not None:
                counts[cons] = int(val)
        logger.info("Resolved dose='all' counts from %s", sanity_path)
    else:
        logger.warning(
            "No dose sanity JSON at %s; using documented fallback counts %s",
            sanity_path, counts,
        )

    out = df.copy()
    numeric_dose = []
    for cons, dose in zip(out["model_construction"], out["dose"], strict=True):
        if str(dose).strip().lower() == "all":
            if cons not in counts:
                raise KeyError(
                    f"No attested-'all' count for construction '{cons}'. "
                    "Add it to ATTESTED_ALL_COUNTS or the sanity JSON."
                )
            numeric_dose.append(float(counts[cons]))
        else:
            numeric_dose.append(float(dose))
    out["dose_value"] = np.asarray(numeric_dose, dtype=float)
    return out


def _initial_guess(D: np.ndarray, Y: np.ndarray) -> list[float]:
    """A sane starting point for the optimiser, read straight off the data."""
    import numpy as np

    E0_0 = float(np.clip(Y[np.argmin(D)], 0.0, 1.0))
    Emax_0 = float(np.clip(Y.max(), E0_0 + 1e-3, 1.0))
    # Halfway dose: the smallest positive dose whose accuracy clears the midpoint,
    # falling back to the geometric mean of the dose range.
    mid = (E0_0 + Emax_0) / 2.0
    pos = D[D > 0]
    above = D[(D > 0) & (Y >= mid)]
    if above.size:
        E50_0 = float(above.min())
    elif pos.size:
        E50_0 = float(np.exp(np.log(pos).mean()))
    else:
        E50_0 = 1.0
    return [E0_0, Emax_0, max(E50_0, 1e-3), 1.0]


def _r_squared(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    import numpy as np

    ss_res = float(np.sum((Y - Y_hat) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    if ss_tot <= 0:  # flat target — R^2 is undefined, report 0
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_one(
    D: np.ndarray, Y: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    """Fit one Hill curve and bootstrap its parameter CIs.

    Returns a flat dict ready to become one CSV row. ``converged`` is False when
    the optimiser raises rather than crashing the whole sweep — one bad cell
    shouldn't lose you the other fifty-nine.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    # E0, Emax bounded to the accuracy range; E50 and n strictly positive.
    lower = [0.0, 0.0, 1e-6, 1e-3]
    upper = [1.0, 1.0, np.inf, 50.0]
    p0 = _initial_guess(D, Y)

    row: dict[str, Any] = dict.fromkeys(PARAM_NAMES, np.nan)
    for p in PARAM_NAMES:
        row[f"{p}_lo"] = np.nan
        row[f"{p}_hi"] = np.nan
    row["r_squared"] = np.nan
    row["converged"] = False

    try:
        popt, _ = curve_fit(
            hill, D, Y, p0=p0, bounds=(lower, upper),
            method="trf", maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001 - record failure, keep going
        logger.warning("Hill fit failed to converge: %s", exc)
        return row

    Y_hat = hill(D, *popt)
    resid = Y - Y_hat
    for name, val in zip(PARAM_NAMES, popt, strict=True):
        row[name] = float(val)
    row["r_squared"] = _r_squared(Y, Y_hat)
    row["converged"] = True

    # Parametric bootstrap: re-draw residuals (with replacement) onto the fitted
    # curve, refit, and keep the parameter draws that converge.
    draws: list[np.ndarray] = []
    for _ in range(N_BOOTSTRAP):
        Y_star = Y_hat + rng.choice(resid, size=resid.size, replace=True)
        Y_star = np.clip(Y_star, 0.0, 1.0)
        try:
            p_star, _ = curve_fit(
                hill, D, Y_star, p0=popt, bounds=(lower, upper),
                method="trf", maxfev=20000,
            )
            draws.append(p_star)
        except Exception:  # noqa: BLE001 - skip non-converging resamples
            continue

    if draws:
        arr = np.vstack(draws)
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        for i, name in enumerate(PARAM_NAMES):
            row[f"{name}_lo"] = float(lo[i])
            row[f"{name}_hi"] = float(hi[i])
    return row


def fit_all(eval_csv: Path, sanity_path: Path | None, seed: int = 0) -> pd.DataFrame:
    """Fit a Hill curve per (construction, seed) over the whole eval table."""
    import numpy as np
    import pandas as pd

    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Eval results not found at {eval_csv}. Run the eval stage first."
        )

    df = pd.read_csv(eval_csv)
    # Self-evaluation only: a construction's curve uses its own minimal pairs.
    df = df[df["model_construction"] == df["eval_construction"]].copy()
    df = resolve_dose_values(df, sanity_path)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for cons in CONSTRUCTIONS:
        for s in sorted(df.loc[df["model_construction"] == cons, "seed"].unique()):
            cell = df[(df["model_construction"] == cons) & (df["seed"] == s)]
            cell = cell.sort_values("dose_value")
            if len(cell) < len(PARAM_NAMES):
                logger.warning(
                    "Skipping %s seed %s: only %d dose points, need >= %d.",
                    cons, s, len(cell), len(PARAM_NAMES),
                )
                continue
            D = cell["dose_value"].to_numpy(dtype=float)
            Y = cell["accuracy"].to_numpy(dtype=float)
            fit = fit_one(D, Y, rng)
            rows.append({"construction": cons, "seed": int(s), **fit})

    if not rows:
        raise RuntimeError(
            "No (construction, seed) cells had enough dose points to fit. "
            "Check that the eval stage wrote all five dose levels."
        )
    return pd.DataFrame(rows)


def run(config_path: Path, seed: int = 0) -> Path:
    """Fit every cell and write ``results/hill_fits.csv``."""
    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    eval_csv = results_dir / "eval_results.csv"
    sanity_path = results_dir / "dose_corpora_sanity.json"

    fits = fit_all(eval_csv, sanity_path, seed=seed)

    out_path = results_dir / "hill_fits.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.to_csv(out_path, index=False)
    logger.info("Wrote %d Hill fits to %s", len(fits), out_path)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the bootstrap RNG (fit results are deterministic given it).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
