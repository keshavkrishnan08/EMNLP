"""Measure collateral dose-response: does dosing one construction move another?

Every model in the sweep is evaluated on all four minimal-pair sets, not just
its own. That cross-eval table is usually treated as a sanity check, but it
hides a question nobody's answered: when you pour exposures into construction
``C``, does the model's acceptability for a *different* construction ``C'`` move
along with it? Call it collateral dose-response. A positive off-diagonal slope
means training on ``C`` quietly helps ``C'``; negative means it hurts.

For each ordered pair ``(model_construction=C, eval_construction=C')`` we fit a
single line: ``C'`` accuracy regressed on ``log10(dose + 1)``, pooling all the
dose levels and seeds into one ordinary least-squares fit. The slope is the
number we keep. The diagonal (``C == C'``) is the familiar on-target
dose-response slope; everything off it is transfer.

We log-transform the dose because the doses are logarithmically spaced (0, 4,
16, 64, all) — a straight line in raw dose would be dominated by the top end.
The ``+1`` keeps dose 0 finite. We resolve ``"all"`` to the construction's
attested count the same way ``hill.py`` does, so the x-axis is consistent with
the curve fits.

Output is a 4x4 slope matrix (rows = trained-on, cols = evaluated-on) plus the
standard error of each slope, and a heatmap with a diverging colormap centred
at zero so "helps" and "hurts" read at a glance.

CLI::

    python -m drc.analysis.transfer --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drc import CONSTRUCTIONS
from drc.analysis.hill import resolve_dose_values
from drc.data.download import load_config, resolve_path

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps numpy/pandas lazy
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


def fit_slope(dose_value: np.ndarray, accuracy: np.ndarray) -> tuple[float, float]:
    """OLS slope of accuracy on ``log10(dose + 1)``, with its standard error.

    Returns ``(slope, stderr)``. We do the algebra by hand rather than pulling
    in statsmodels: it's a one-predictor regression, the closed form is three
    lines, and it keeps the dependency surface small. NaN comes back when there
    aren't enough distinct dose points to define a line.
    """
    import numpy as np

    x = np.log10(np.asarray(dose_value, dtype=float) + 1.0)
    y = np.asarray(accuracy, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")

    x_mean = x.mean()
    sxx = float(np.sum((x - x_mean) ** 2))
    slope = float(np.sum((x - x_mean) * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * x_mean)

    resid = y - (intercept + slope * x)
    dof = n - 2
    if dof <= 0:
        return slope, float("nan")
    sigma2 = float(np.sum(resid ** 2) / dof)
    stderr = float(np.sqrt(sigma2 / sxx))
    return slope, stderr


def build_transfer(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a transfer slope for every ordered (trained-on, evaluated-on) pair.

    Expects the *full* eval table with ``dose_value`` already resolved. Returns
    ``(slopes, stderrs)``, both 4x4 DataFrames indexed by trained-on
    construction with columns the evaluated-on construction. A missing pair
    (no rows) yields NaN rather than zero, so an absent cell can't masquerade
    as "no transfer".
    """
    import numpy as np
    import pandas as pd

    slopes = pd.DataFrame(
        np.full((len(CONSTRUCTIONS), len(CONSTRUCTIONS)), np.nan),
        index=list(CONSTRUCTIONS), columns=list(CONSTRUCTIONS), dtype=float,
    )
    stderrs = slopes.copy()

    for train in CONSTRUCTIONS:
        for ev in CONSTRUCTIONS:
            cell = eval_df[
                (eval_df["model_construction"] == train)
                & (eval_df["eval_construction"] == ev)
            ]
            if cell.empty:
                logger.warning("No rows for trained=%s evaluated=%s.", train, ev)
                continue
            slope, stderr = fit_slope(
                cell["dose_value"].to_numpy(dtype=float),
                cell["accuracy"].to_numpy(dtype=float),
            )
            slopes.loc[train, ev] = slope
            stderrs.loc[train, ev] = stderr

    return slopes, stderrs


def _tidy_matrix(slopes: pd.DataFrame, stderrs: pd.DataFrame) -> pd.DataFrame:
    """Flatten the two matrices into one tidy CSV-friendly table.

    Long form (one row per pair) plus an ``is_diagonal`` flag travels better
    than a wide grid: it's trivial to filter to on-target vs transfer in a
    spreadsheet, and it keeps the slope and its stderr side by side.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for train in CONSTRUCTIONS:
        for ev in CONSTRUCTIONS:
            rows.append(
                {
                    "model_construction": train,
                    "eval_construction": ev,
                    "slope": float(slopes.loc[train, ev]),
                    "stderr": float(stderrs.loc[train, ev]),
                    "is_diagonal": train == ev,
                }
            )
    return pd.DataFrame(rows)


def fig7_transfer_matrix(slopes: pd.DataFrame, out_path: Path) -> None:
    """4x4 heatmap of transfer slopes, diverging map centred at zero.

    Rows are the trained-on construction, columns the evaluated-on one. The
    diagonal — the on-target dose-response — is boxed so it reads apart from the
    collateral cells. Colour runs blue-white-red through zero (RdBu_r), so a
    slope of zero is white: no movement, no colour. Cells are annotated with the
    slope value.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from drc.analysis.figures import DISPLAY_NAMES

    mat = slopes.to_numpy(dtype=float)
    finite = mat[np.isfinite(mat)]
    # Symmetric colour limits so zero sits dead centre on white.
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    labels = [DISPLAY_NAMES.get(c, c) for c in CONSTRUCTIONS]
    ax.set_xticks(range(len(CONSTRUCTIONS)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(CONSTRUCTIONS)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Evaluated on")
    ax.set_ylabel("Trained on (dosed)")
    ax.set_title("Cross-construction transfer: slope of accuracy vs log dose")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("OLS slope (Δ accuracy per decade of dose)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i, j]):
                continue
            # White text on saturated cells, black near the neutral centre.
            strong = abs(mat[i, j]) > 0.5 * vmax
            ax.text(
                j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                color="white" if strong else "black", fontsize=8,
            )
            if i == j:  # box the on-target diagonal
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor="black", lw=2.0,
                    )
                )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def run(config_path: Path) -> dict[str, Path]:
    """Fit the transfer matrix from the full eval table; write CSV + heatmap."""
    import pandas as pd

    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    eval_csv = results_dir / "eval_results.csv"
    sanity_path = results_dir / "dose_corpora_sanity.json"

    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Eval results not found at {eval_csv}. Run the eval stage first."
        )

    eval_df = pd.read_csv(eval_csv)
    eval_df = resolve_dose_values(eval_df, sanity_path)
    slopes, stderrs = build_transfer(eval_df)

    results_dir.mkdir(parents=True, exist_ok=True)
    tidy = _tidy_matrix(slopes, stderrs)
    out_csv = results_dir / "transfer_matrix.csv"
    tidy.to_csv(out_csv, index=False)
    logger.info("Wrote %d transfer pairs to %s", len(tidy), out_csv)

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "fig7_transfer_matrix.pdf"
    fig7_transfer_matrix(slopes, fig_path)

    return {"matrix": out_csv, "figure": fig_path}


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
