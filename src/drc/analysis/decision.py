"""Apply the pre-registered decision rules and name the winning hypothesis.

We locked these five rules before looking at the fitted numbers, which is the
whole point — the result this script prints is whatever the data says, not a
story we backfilled. Each rule maps to a paper title from the PRD Outcome
Matrix, so the output doubles as "here's what we're calling the paper."

The rules, in the order we check them:

* **H1 Smooth Scaling** — Hill wins by AIC for every construction, and the mean
  Hill coefficient sits in [0.5, 2.0]. Learning is a gentle saturating climb.
* **H2 Phase Transition** — Hill wins everywhere, but mean n > 4. Same curve,
  far steeper: grammatical grokking.
* **H3 Heterogeneous** — k=2 clusters separate cleanly (silhouette > 0.5), or
  Hill wins for some constructions while step wins for others. Two routes.
* **H4 Indirect Evidence Dominates** — the climb from floor to ceiling is tiny
  (mean (Emax-E0)/Emax < 0.1). Direct exposure barely moves the needle.
* **H5 Stochastic** — seed-to-seed wobble swamps the dose effect: mean
  within-cell CV across seeds exceeds mean between-cell CV across doses.

If more than one fires, or none does cleanly, we say "Mixed" and point back to
the matrix rather than forcing a headline.

CLI::

    python -m drc.analysis.decision --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drc import CONSTRUCTIONS
from drc.data.download import load_config, resolve_path

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = logging.getLogger(__name__)

# Hypothesis -> paper title, straight from the PRD Outcome Matrix.
TITLES: dict[str, str] = {
    "H1": "A Scaling Law for Construction Learning in Language Models",
    "H2": "Grammatical Grokking: Phase Transitions in Construction Acquisition",
    "H3": "Two Routes to Grammar: Heterogeneous Dose-Response in Construction Learning",
    "H4": "Direct Exposure Doesn't Matter: Indirect Evidence in Construction Learning",
    "H5": "Stochastic Construction Acquisition: When Seeds Beat Doses",
}

# Thresholds, pinned here so they read as the pre-registered constants they are.
HILL_N_SMOOTH = (0.5, 2.0)
HILL_N_PHASE = 4.0
SILHOUETTE_SPLIT = 0.5
INDIRECT_GAIN_MAX = 0.1


def _best_model_by_construction(model_comp: pd.DataFrame) -> dict[str, str]:
    """For each construction, which model has the lowest mean AIC across seeds."""
    converged = model_comp[model_comp["converged"].astype(bool)]
    best: dict[str, str] = {}
    for cons in CONSTRUCTIONS:
        sub = converged[converged["construction"] == cons]
        if sub.empty:
            continue
        mean_aic = sub.groupby("model")["aic"].mean()
        best[cons] = str(mean_aic.idxmin())
    return best


def _within_vs_between_cv(eval_df: pd.DataFrame) -> tuple[float, float]:
    """Compare seed noise to dose signal via coefficients of variation.

    Within-cell CV: for a fixed (construction, dose), how much does accuracy
    wobble across seeds? Between-cell CV: for a fixed (construction, seed), how
    much does accuracy move across doses? If the first beats the second, the
    model's grammar is basically a coin flip on the random seed (H5).
    """
    import numpy as np

    def cv(values) -> float:
        v = np.asarray(values, dtype=float)
        m = v.mean()
        return float(v.std() / m) if m != 0 else 0.0

    within = (
        eval_df.groupby(["model_construction", "dose"])["accuracy"]
        .apply(cv)
    )
    between = (
        eval_df.groupby(["model_construction", "seed"])["accuracy"]
        .apply(cv)
    )
    return float(within.mean()), float(between.mean())


def evaluate(
    hill_fits: pd.DataFrame,
    model_comp: pd.DataFrame,
    clusters: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute every rule's inputs and decide which hypotheses fire.

    Returns a dict of the supporting statistics plus the list of fired
    hypotheses, so the writer below can both decide and explain.
    """
    import numpy as np

    good = hill_fits[hill_fits["converged"].astype(bool)]
    mean_n = float(good["n"].mean()) if not good.empty else float("nan")

    best_model = _best_model_by_construction(model_comp)
    hill_best_all = bool(best_model) and all(
        best_model.get(c) == "hill" for c in CONSTRUCTIONS
    )
    some_hill = any(v == "hill" for v in best_model.values())
    some_step = any(v == "step" for v in best_model.values())

    # k=2 silhouette (constant within the k=2 column).
    sil_k2 = float("nan")
    if "silhouette_k2" in clusters.columns and not clusters.empty:
        col = clusters["silhouette_k2"].dropna()
        if not col.empty:
            sil_k2 = float(col.iloc[0])

    # Mean relative gain from floor to ceiling.
    rel_gain = float("nan")
    if not good.empty:
        gain = (good["Emax"] - good["E0"]) / good["Emax"].replace(0, np.nan)
        rel_gain = float(gain.mean())

    within_cv, between_cv = _within_vs_between_cv(eval_df)

    fired: list[str] = []
    if hill_best_all and HILL_N_SMOOTH[0] <= mean_n <= HILL_N_SMOOTH[1]:
        fired.append("H1")
    if hill_best_all and mean_n > HILL_N_PHASE:
        fired.append("H2")
    if (not np.isnan(sil_k2) and sil_k2 > SILHOUETTE_SPLIT) or (some_hill and some_step):
        fired.append("H3")
    if not np.isnan(rel_gain) and rel_gain < INDIRECT_GAIN_MAX:
        fired.append("H4")
    if within_cv > between_cv:
        fired.append("H5")

    return {
        "mean_n": mean_n,
        "best_model": best_model,
        "hill_best_all": hill_best_all,
        "silhouette_k2": sil_k2,
        "rel_gain": rel_gain,
        "within_cv": within_cv,
        "between_cv": between_cv,
        "fired": fired,
    }


def render(stats: dict[str, Any]) -> str:
    """Turn the computed stats into the human-readable decision report."""
    fired = stats["fired"]
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("DRC PRE-REGISTERED DECISION")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "NOTE: every number below is computed from the fitted curves and raw "
        "eval CSVs. Nothing here is predetermined — the hypothesis that fires "
        "is whatever the data supports under the rules we locked in advance."
    )
    lines.append("")

    lines.append("Supporting statistics")
    lines.append("-" * 70)
    lines.append(f"  mean Hill coefficient n      : {stats['mean_n']:.3f}")
    lines.append(f"  Hill best by AIC for all?    : {stats['hill_best_all']}")
    lines.append(f"  best model per construction  : {stats['best_model']}")
    lines.append(f"  k=2 silhouette               : {stats['silhouette_k2']:.3f}")
    lines.append(f"  mean (Emax-E0)/Emax          : {stats['rel_gain']:.3f}")
    lines.append(f"  mean within-cell CV (seeds)  : {stats['within_cv']:.4f}")
    lines.append(f"  mean between-cell CV (doses) : {stats['between_cv']:.4f}")
    lines.append("")

    lines.append("Decision")
    lines.append("-" * 70)
    if len(fired) == 1:
        h = fired[0]
        lines.append(f"  Hypothesis fired : {h}")
        lines.append(f"  Recommended title: {TITLES[h]}")
    elif len(fired) == 0:
        lines.append("  Hypothesis fired : none cleanly")
        lines.append("  Recommended title: Mixed — see Outcome Matrix")
    else:
        lines.append(f"  Hypotheses fired : {', '.join(fired)} (more than one)")
        lines.append("  Recommended title: Mixed — see Outcome Matrix")
        lines.append("  Candidate titles for the ones that fired:")
        for h in fired:
            lines.append(f"    {h}: {TITLES[h]}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


def run(config_path: Path) -> Path:
    """Load every analysis output, decide, and write ``results/decision.txt``."""
    import pandas as pd

    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])

    needed = {
        "hill_fits.csv": results_dir / "hill_fits.csv",
        "model_comparison.csv": results_dir / "model_comparison.csv",
        "cluster_assignments.csv": results_dir / "cluster_assignments.csv",
        "eval_results.csv": results_dir / "eval_results.csv",
    }
    for name, path in needed.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} at {path}. Run the earlier analysis stages first."
            )

    hill_fits = pd.read_csv(needed["hill_fits.csv"])
    model_comp = pd.read_csv(needed["model_comparison.csv"])
    clusters = pd.read_csv(needed["cluster_assignments.csv"])
    eval_df = pd.read_csv(needed["eval_results.csv"])
    eval_df = eval_df[eval_df["model_construction"] == eval_df["eval_construction"]].copy()

    stats = evaluate(hill_fits, model_comp, clusters, eval_df)
    report = render(stats)

    out_path = results_dir / "decision.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    logger.info("Wrote decision to %s", out_path)
    # Echo to the log so a sweep run shows the verdict without opening the file.
    for line in report.splitlines():
        logger.info("%s", line)
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
