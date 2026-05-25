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

Tiered design (see ``drc.design``). The sweep no longer trains a full dose
ladder for every construction:

* **Core** constructions get the intermediate doses ``[0, 4, 16, 64]`` plus the
  shared ceiling, so they have >= 4 dose points and we fit the full Hill curve
  as before.
* **Breadth** constructions get only ``{0, all}`` --- two points. Two points
  can't pin down four Hill parameters, so we do NOT fit. Instead we record the
  measured ``E0`` (dose-0 accuracy) and ``Emax`` (the shared-ceiling accuracy)
  directly, leave ``E50``/``n`` as NaN, and set ``converged=False``. The
  ``hill_fits.csv`` schema is unchanged; breadth rows just carry NaN where a fit
  would have gone.

The ceiling point for *every* construction comes from the one shared
full-corpus model (``model_construction == "full"``, ``dose == "all"``),
evaluated on that construction's minimal pairs. That row supplies the ``"all"``
point of the series. See ``drc.design.emax_construction``.

CLI::

    python -m drc.analysis.hill --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drc.data.download import load_config, resolve_path
from drc.design import (
    all_constructions,
    curve_doses,
    emax_construction,
    is_core,
)

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
# corpus regeneration. Only the *core* constructions need a numeric "all" dose
# (it's the x-position of the ceiling point in the curve fit); breadth
# constructions are never fit, so a missing count for them is harmless.
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


def load_all_counts(sanity_path: Path | None) -> dict[str, int]:
    """Per-construction attested 'all' counts, from the sanity JSON or fallback.

    ``"all"`` means "every instance we found in the corpus", and the actual
    number differs per construction. We read it from
    ``results/dose_corpora_sanity.json`` when present, and from the documented
    fallback table otherwise.
    """
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
    return counts


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

    counts = load_all_counts(sanity_path)

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


def _empty_fit_row() -> dict[str, Any]:
    """A fit row pre-filled with NaNs and ``converged=False``.

    Used as the template for breadth constructions (no Hill fit) so every row in
    ``hill_fits.csv`` carries the same columns regardless of tier.
    """
    import numpy as np

    row: dict[str, Any] = dict.fromkeys(PARAM_NAMES, np.nan)
    for p in PARAM_NAMES:
        row[f"{p}_lo"] = np.nan
        row[f"{p}_hi"] = np.nan
    row["r_squared"] = np.nan
    row["converged"] = False
    return row


def construction_series(
    df: pd.DataFrame,
    config: dict[str, Any],
    cons: str,
    seed: int,
    counts: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble one construction's (dose_value, accuracy) series for a seed.

    Walks the design's ``curve_doses`` for the construction. For every dose
    except ``"all"`` it reads the self-eval row of that construction's own model
    at that dose. For ``"all"`` it reads the *shared ceiling* model's row
    (``model_construction == emax_construction(config, cons)``, i.e. ``"full"``)
    evaluated on this construction. The numeric x-value of ``"all"`` is the
    construction's attested count. Doses with no matching eval row are skipped.

    The numeric ``"all"`` x-value is only meaningful for *core* constructions,
    which we actually fit; for breadth constructions the curve is never fit, so
    a missing attested count falls back to ``+inf`` (it keeps the ceiling last in
    the sort and never feeds an optimiser).
    """
    import numpy as np

    self_rows = df[(df["eval_construction"] == cons) & (df["seed"] == seed)]
    ceiling_code = emax_construction(config, cons)
    core = is_core(config, cons)
    doses_x: list[float] = []
    accs: list[float] = []
    for dose in curve_doses(config, cons):
        if str(dose).strip().lower() == "all":
            row = self_rows[
                (self_rows["model_construction"] == ceiling_code)
                & (self_rows["dose"].astype(str).str.strip().str.lower() == "all")
            ]
            if row.empty:
                logger.warning(
                    "No ceiling row for %s (model_construction==%s, dose=all).",
                    cons, ceiling_code,
                )
                continue
            if cons not in counts:
                if core:
                    raise KeyError(
                        f"No attested-'all' count for core construction '{cons}'. "
                        "Add it to ATTESTED_ALL_COUNTS or the sanity JSON."
                    )
                doses_x.append(float("inf"))  # breadth: x unused, keep last
            else:
                doses_x.append(float(counts[cons]))
            accs.append(float(row["accuracy"].iloc[0]))
        else:
            row = self_rows[
                (self_rows["model_construction"] == cons)
                & (self_rows["dose"].astype(str).str.strip() == str(dose))
            ]
            if row.empty:
                continue
            doses_x.append(float(dose))
            accs.append(float(row["accuracy"].iloc[0]))
    order = np.argsort(np.asarray(doses_x, dtype=float))
    return (
        np.asarray(doses_x, dtype=float)[order],
        np.asarray(accs, dtype=float)[order],
    )


def fit_all(
    eval_csv: Path,
    sanity_path: Path | None,
    config: dict[str, Any],
    seed: int = 0,
) -> pd.DataFrame:
    """Build each construction's dose series and fit (core) or record (breadth).

    Core constructions (>= 4 dose points) get the four-parameter Hill fit.
    Breadth constructions (only dose 0 + the shared ceiling) can't be fit, so we
    record their measured ``E0`` and ``Emax`` directly and leave ``E50``/``n``
    NaN with ``converged=False``. The ceiling point for every construction comes
    from the shared full-corpus model (see ``construction_series``).
    """
    import numpy as np
    import pandas as pd

    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Eval results not found at {eval_csv}. Run the eval stage first."
        )

    df = pd.read_csv(eval_csv)
    counts = load_all_counts(sanity_path)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for cons in all_constructions(config):
        core = is_core(config, cons)
        for s in sorted(df.loc[df["seed"].notna(), "seed"].astype(int).unique()):
            D, Y = construction_series(df, config, cons, int(s), counts)
            if D.size == 0:
                continue
            if core:
                if D.size < len(PARAM_NAMES):
                    logger.warning(
                        "Skipping core %s seed %s: only %d dose points, need >= %d.",
                        cons, s, D.size, len(PARAM_NAMES),
                    )
                    continue
                fit = fit_one(D, Y, rng)
            else:
                # Breadth: no fit. Record E0 (dose-0 accuracy) and Emax (ceiling).
                fit = _empty_fit_row()
                fit["E0"] = float(Y[0])
                fit["Emax"] = float(Y[-1])
            rows.append({"construction": cons, "seed": int(s), "is_core": core, **fit})

    if not rows:
        raise RuntimeError(
            "No (construction, seed) cells had a usable dose series. Check the "
            "eval stage wrote per-dose rows and a model_construction=='full' row."
        )
    return pd.DataFrame(rows)


def run(config_path: Path, seed: int = 0) -> Path:
    """Fit every cell and write ``results/hill_fits.csv``."""
    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    eval_csv = results_dir / "eval_results.csv"
    sanity_path = results_dir / "dose_corpora_sanity.json"

    fits = fit_all(eval_csv, sanity_path, config, seed=seed)

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
