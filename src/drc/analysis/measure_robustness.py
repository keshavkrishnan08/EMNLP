"""Does the dose-response survive a change of acceptability measure?

A reviewer can fairly worry that the whole story rides on SLOR. Maybe the curves
are an artefact of subtracting a unigram baseline, not a real fact about what the
models learned. This module answers that with zero extra training: it re-scores
the same minimal pairs under a second, simpler measure — the plain
length-normalised pseudo-log-likelihood (mean-LP), with no unigram correction —
which ``run_eval`` already recorded alongside SLOR for free.

The check is deliberately blunt. For every ``(construction, dose)`` cell we
compute accuracy two ways: SLOR via the existing ``correct`` flag, mean-LP via
``correct_meanlp``. Then we ask three questions:

* Do the two per-cell accuracies *correlate* (Pearson and Spearman)? If the
  curves move together, the metric choice is not driving the result.
* Is the per-construction ordering of the indirect-evidence floor ``E0``
  (dose-0 accuracy) preserved? Constructions that score highest at zero exposure
  under SLOR should also score highest under mean-LP — a rank correlation.
* How far apart are the two measures on average (mean absolute accuracy
  difference)? Small means they barely disagree.

We lean on ``design`` exactly like ``hill.py`` does, so the dose=='all' ceiling
point comes from the shared full-corpus model (``model_construction == "full"``)
and not from a per-construction run. The figure overlays the two dose-response
curves per core construction over ``log10(dose + 1)``.

CLI::

    python -m drc.analysis.measure_robustness --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drc.data.download import load_config, resolve_path
from drc.design import core_constructions, curve_doses, emax_construction

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps numpy/pandas lazy
    import pandas as pd

logger = logging.getLogger(__name__)


def _require_per_item(per_item_csv: Path) -> None:
    """Stop loudly when the per-item log is missing — there's nothing to check.

    The robustness check is built entirely on ``eval_per_item.csv`` (it needs
    the two correctness flags per pair). Point the caller at the eval stage
    rather than letting a downstream read blow up with a vaguer error.
    """
    if not per_item_csv.exists():
        raise FileNotFoundError(
            f"Per-item eval log not found at {per_item_csv}. Run the eval stage "
            "first (python -m drc.eval.run_eval) — it writes eval_per_item.csv "
            "with the `correct` and `correct_meanlp` columns this check needs."
        )


def compute(per_item_csv: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Build the per-(construction, dose) accuracy table under both measures.

    A "cell" is one construction's own model at one dose, scored on that same
    construction's minimal pairs, pooled over every seed and item. We take the
    self-eval rows (``model_construction == eval_construction``) so a cell
    measures direct learning. The dose=='all' ceiling resolves through
    ``design.emax_construction`` to the shared full-corpus model — the same path
    ``hill.py`` takes — so the two stay consistent about where the ceiling comes
    from. Everything heavy (pandas) is imported inside the function.

    Returns a long table with one row per (construction, dose) carrying
    ``acc_slor``, ``acc_meanlp`` and ``n`` (items pooled across seeds).
    """
    import pandas as pd

    _require_per_item(per_item_csv)
    per_item = pd.read_csv(per_item_csv)

    missing = [c for c in ("correct", "correct_meanlp") if c not in per_item.columns]
    if missing:
        raise KeyError(
            f"eval_per_item.csv is missing column(s) {missing}. Re-run the eval "
            "stage — older runs predate the mean-LP columns."
        )

    df = per_item.copy()
    for col in ("correct", "correct_meanlp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    dose_str = df["dose"].astype(str).str.strip().str.lower()

    rows: list[dict[str, Any]] = []
    for cons in core_constructions(config):
        ceiling_code = emax_construction(config, cons)
        for dose in curve_doses(config, cons):
            if str(dose).strip().lower() == "all":
                mask = (
                    (df["eval_construction"] == cons)
                    & (df["model_construction"] == ceiling_code)
                    & (dose_str == "all")
                )
            else:
                mask = (
                    (df["eval_construction"] == cons)
                    & (df["model_construction"] == cons)
                    & (dose_str == str(dose).strip())
                )
            cell = df[mask]
            if cell.empty:
                logger.warning("No per-item rows for %s dose=%s; skipping.", cons, dose)
                continue
            rows.append(
                {
                    "construction": cons,
                    "dose": str(dose),
                    "acc_slor": float(cell["correct"].mean()),
                    "acc_meanlp": float(cell["correct_meanlp"].mean()),
                    "n": int(cell["correct"].notna().sum()),
                }
            )

    if not rows:
        raise RuntimeError(
            "No (construction, dose) cells had per-item rows. Check that the eval "
            "stage wrote eval_per_item.csv with a model_construction=='full' "
            "ceiling row and per-dose self-eval rows."
        )
    return pd.DataFrame(rows)


def agreement_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Quantify how closely the two measures agree across cells.

    Three numbers, each a one-row entry in the returned table:

    * ``pearson`` / ``spearman`` — correlation of the per-cell SLOR and mean-LP
      accuracies. High means the curves rise and fall together.
    * ``e0_rank_spearman`` — Spearman correlation of the per-construction
      dose-0 accuracy *ranking* between the two measures. This is the construct
      that matters most for the indirect-evidence story: does the ordering of
      who-learns-what-from-nothing survive the metric swap?
    * ``mean_abs_diff`` — average ``|acc_slor - acc_meanlp|`` over cells. A small
      value means the measures rarely disagree on the raw number, not just the
      ranking.

    Correlations need at least two distinct points; with fewer we record NaN
    rather than a misleading 1.0.
    """
    import numpy as np
    import pandas as pd

    slor = cells["acc_slor"].to_numpy(dtype=float)
    meanlp = cells["acc_meanlp"].to_numpy(dtype=float)

    pearson = _safe_corr(slor, meanlp, kind="pearson")
    spearman = _safe_corr(slor, meanlp, kind="spearman")

    # E0 ordering: one dose-0 accuracy per construction, under each measure.
    zero = cells[cells["dose"].astype(str).str.strip().str.lower() == "0"]
    zero = zero.sort_values("construction")
    e0_rank = _safe_corr(
        zero["acc_slor"].to_numpy(dtype=float),
        zero["acc_meanlp"].to_numpy(dtype=float),
        kind="spearman",
    )

    mean_abs_diff = float(np.mean(np.abs(slor - meanlp))) if slor.size else float("nan")

    return pd.DataFrame(
        [
            {
                "n_cells": int(cells.shape[0]),
                "n_constructions_e0": int(zero.shape[0]),
                "pearson": pearson,
                "spearman": spearman,
                "e0_rank_spearman": e0_rank,
                "mean_abs_diff": mean_abs_diff,
            }
        ]
    )


def _safe_corr(a, b, *, kind: str) -> float:
    """Correlation that returns NaN instead of throwing on degenerate input.

    SciPy warns and returns NaN when a vector is constant (zero variance), and
    raises on fewer than two points. We swallow both so a thin synthetic table
    or a flat construction doesn't crash the whole stage — NaN is the honest
    answer when a correlation isn't defined.
    """
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    from scipy import stats

    if kind == "pearson":
        r = stats.pearsonr(a, b)[0]
    else:
        r = stats.spearmanr(a, b)[0]
    return float(r)


def fig10_measure_robustness(cells: pd.DataFrame, config: dict[str, Any], out_path: Path) -> None:
    """2x2 panel: SLOR vs mean-LP dose-response, one panel per core construction.

    Each panel overlays the two measures' accuracy curves over
    ``log10(dose + 1)``, using the construction's attested 'all' count for the
    ceiling x-position via the shared dose resolver. Okabe-Ito colours and the
    spare publication theme, borrowed straight from ``figures.py`` so this figure
    looks like the others.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from drc.analysis.figures import DISPLAY_NAMES, _apply_minimal_theme
    from drc.analysis.hill import load_all_counts

    cons_list = core_constructions(config)
    counts = load_all_counts(None)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    fig.suptitle("Measure robustness: SLOR vs mean-LP dose-response")

    # Pad the construction list out to four panels; hide any spare axes.
    flat = list(axes.flat)
    panels = list(zip(flat, cons_list[:4], strict=False))
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    for ax, cons in panels:
        sub = cells[cells["construction"] == cons].copy()
        if sub.empty:
            ax.set_title(DISPLAY_NAMES.get(cons, cons))
            _apply_minimal_theme(ax)
            continue

        all_x = float(counts.get(cons, np.nan))
        x = sub["dose"].map(
            lambda d, all_x=all_x: all_x
            if str(d).strip().lower() == "all"
            else float(d)
        )
        order = np.argsort(x.to_numpy(dtype=float))
        xlog = np.log10(x.to_numpy(dtype=float)[order] + 1.0)
        ax.plot(
            xlog, sub["acc_slor"].to_numpy(dtype=float)[order],
            color="#E69F00", lw=2.0, marker="o", label="SLOR",
        )
        ax.plot(
            xlog, sub["acc_meanlp"].to_numpy(dtype=float)[order],
            color="#56B4E9", lw=2.0, marker="s", ls="--", label="mean-LP",
        )
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")  # chance
        ax.set_title(DISPLAY_NAMES.get(cons, cons))
        ax.set_xlabel(r"$\log_{10}(\mathrm{dose} + 1)$")
        ax.set_ylabel("Minimal-pair accuracy")
        ax.set_ylim(0.0, 1.0)
        _apply_minimal_theme(ax)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def run(config_path: Path) -> dict[str, Path]:
    """Compute the per-cell table + agreement summary, write both CSVs and fig10."""
    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    per_item_csv = results_dir / "eval_per_item.csv"

    cells = compute(per_item_csv, config)
    summary = agreement_summary(cells)

    results_dir.mkdir(parents=True, exist_ok=True)
    cells_path = results_dir / "measure_robustness.csv"
    summary_path = results_dir / "measure_robustness_summary.csv"
    cells.to_csv(cells_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %d cells to %s", len(cells), cells_path)
    logger.info(
        "Agreement: pearson=%.3f spearman=%.3f E0-rank=%.3f mean|diff|=%.3f",
        summary["pearson"].iloc[0], summary["spearman"].iloc[0],
        summary["e0_rank_spearman"].iloc[0], summary["mean_abs_diff"].iloc[0],
    )

    fig_dir = results_dir / "figures"
    fig_path = fig_dir / "fig10_measure_robustness.pdf"
    fig10_measure_robustness(cells, config, fig_path)

    return {"cells": cells_path, "summary": summary_path, "figure": fig_path}


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
