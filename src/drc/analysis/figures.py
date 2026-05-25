"""Generate the five publication figures, all straight from the results CSVs.

Nothing here is hand-drawn or hard-coded. Each figure reads the eval table (and,
where relevant, the Hill fits and n-gram baseline) and renders to PDF, so a
regenerated sweep produces regenerated figures with zero manual edits. If an
input CSV is missing we raise loudly — a figure built on absent data would be
worse than no figure.

Style is deliberately spare: the Okabe-Ito colorblind-safe palette, top and
right spines off, every axis labelled. That keeps the panels legible in print
and friendly to reviewers who can't distinguish red from green.

The five figures:

1. dose-response curves (2x2): per-seed lines, the mean, a bootstrap ribbon, and
   the fitted Hill overlay — the paper's centrepiece.
2. parameter clusters: E0/E50/n scattered, colour by construction, marker by seed.
3. seed variance: a construction-by-dose heatmap of CV across seeds.
4. n-gram comparison (2x2): the neural LM against the 4-gram control.
5. collateral effects: a 4x4 train-by-eval heatmap of mean accuracy.

CLI::

    python -m drc.analysis.figures --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from drc import CONSTRUCTIONS
from drc.analysis.hill import hill, resolve_dose_values
from drc.data.download import load_config, resolve_path

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# Okabe-Ito palette: orange, sky blue, bluish green, yellow. Colorblind-safe and
# the de-facto standard for accessible scientific figures. One colour per
# construction, in the canonical CONSTRUCTIONS order.
OKABE_ITO = ("#E69F00", "#56B4E9", "#009E73", "#F0E442")

# Readable display names for the four constructions.
DISPLAY_NAMES = {
    "aann": "AANN",
    "comparative_correlative": "Comp. Correlative",
    "tough_movement": "Tough Movement",
    "resultative": "Resultative",
}

# Marker per seed for the cluster scatter, cycled if there are extra seeds.
SEED_MARKERS = ("o", "s", "^", "D", "v", "P")

N_BOOTSTRAP_RIBBON = 1000


def _apply_minimal_theme(ax: Axes) -> None:
    """Strip the top and right spines for the clean publication look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _color_for(cons: str) -> str:
    return OKABE_ITO[CONSTRUCTIONS.index(cons) % len(OKABE_ITO)]


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}. Run the relevant stage before figures."
        )


def _bootstrap_mean_ci(values, n_boot: int, rng):
    """95% CI of the mean by resampling the per-seed values at one dose."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    if v.size < 2:
        m = float(v.mean()) if v.size else float("nan")
        return m, m, m
    draws = np.array([rng.choice(v, size=v.size, replace=True).mean()
                      for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fig1_dose_response_curves(
    eval_df: pd.DataFrame, hill_fits: pd.DataFrame, out_path: Path
) -> None:
    """2x2 dose-response panel: per-seed lines, mean, bootstrap ribbon, Hill fit."""
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    fig.suptitle("Dose-response curves by construction")

    for ax, cons in zip(axes.flat, CONSTRUCTIONS, strict=True):
        sub = eval_df[eval_df["model_construction"] == cons]
        color = _color_for(cons)

        # Per-seed thin lines.
        for s in sorted(sub["seed"].unique()):
            cell = sub[sub["seed"] == s].sort_values("dose_value")
            x = np.log10(cell["dose_value"].to_numpy(dtype=float) + 1.0)
            ax.plot(x, cell["accuracy"], color=color, alpha=0.3, lw=0.8)

        # Mean and bootstrap ribbon across seeds, per dose.
        doses = sorted(sub["dose_value"].unique())
        means, los, his = [], [], []
        for d in doses:
            accs = sub.loc[sub["dose_value"] == d, "accuracy"].to_numpy(dtype=float)
            m, lo, hi = _bootstrap_mean_ci(accs, N_BOOTSTRAP_RIBBON, rng)
            means.append(m)
            los.append(lo)
            his.append(hi)
        xd = np.log10(np.asarray(doses, dtype=float) + 1.0)
        ax.fill_between(xd, los, his, color=color, alpha=0.2, linewidth=0)
        ax.plot(xd, means, color=color, lw=2.0, marker="o", label="mean")

        # Hill overlay: average the converged fits for this construction, then
        # draw the curve on a dense dose grid.
        cons_fits = hill_fits[
            (hill_fits["construction"] == cons) & (hill_fits["converged"].astype(bool))
        ]
        if not cons_fits.empty:
            E0, Emax, E50, n = (
                cons_fits["E0"].mean(), cons_fits["Emax"].mean(),
                cons_fits["E50"].mean(), cons_fits["n"].mean(),
            )
            grid = np.logspace(0, np.log10(max(doses) + 1.0), 200) - 1.0
            grid = np.clip(grid, 0.0, None)
            ax.plot(np.log10(grid + 1.0), hill(grid, E0, Emax, E50, n),
                    color="black", lw=1.2, ls="--", label="Hill fit")

        ax.axhline(0.5, color="grey", lw=0.6, ls=":")  # chance line
        ax.set_title(DISPLAY_NAMES.get(cons, cons))
        ax.set_xlabel(r"$\log_{10}(\mathrm{dose} + 1)$")
        ax.set_ylabel("SLOR accuracy")
        ax.set_ylim(0.0, 1.0)
        _apply_minimal_theme(ax)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def fig2_parameter_clusters(hill_fits: pd.DataFrame, out_path: Path) -> None:
    """Scatter of E0 vs n with E50 as marker size; colour by construction, shape by seed."""
    import matplotlib.pyplot as plt
    import numpy as np

    good = hill_fits[hill_fits["converged"].astype(bool)]
    if good.empty:
        raise RuntimeError("No converged Hill fits to plot for fig2.")

    fig, ax = plt.subplots(figsize=(7, 6))
    seeds = sorted(good["seed"].unique())
    # Size encodes E50 on a log scale so order-of-magnitude differences read.
    e50 = good["E50"].to_numpy(dtype=float)
    sizes = 40 + 120 * (np.log10(e50 + 1.0) / np.log10(e50.max() + 1.0 + 1e-9))

    for cons in CONSTRUCTIONS:
        for j, s in enumerate(seeds):
            cell = good[(good["construction"] == cons) & (good["seed"] == s)]
            if cell.empty:
                continue
            idx = cell.index
            ax.scatter(
                cell["E0"], cell["n"],
                s=[sizes[good.index.get_loc(i)] for i in idx],
                color=_color_for(cons),
                marker=SEED_MARKERS[j % len(SEED_MARKERS)],
                edgecolor="black", linewidth=0.4, alpha=0.85,
            )

    # Two legends: colour=construction, marker=seed.
    from matplotlib.lines import Line2D
    cons_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_color_for(c),
               markersize=8, label=DISPLAY_NAMES.get(c, c))
        for c in CONSTRUCTIONS
    ]
    seed_handles = [
        Line2D([0], [0], marker=SEED_MARKERS[j % len(SEED_MARKERS)], color="w",
               markerfacecolor="grey", markersize=8, label=f"seed {s}")
        for j, s in enumerate(seeds)
    ]
    leg1 = ax.legend(handles=cons_handles, title="Construction",
                     loc="upper right", frameon=False, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=seed_handles, title="Seed (marker size = E50)",
              loc="lower right", frameon=False, fontsize=8)

    ax.set_xlabel(r"$E_0$ (zero-dose accuracy)")
    ax.set_ylabel(r"$n$ (Hill coefficient)")
    ax.set_title("Fitted Hill parameters per construction and seed")
    _apply_minimal_theme(ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def fig3_seed_variance(eval_df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of cross-seed CV: rows are constructions, columns are doses."""
    import matplotlib.pyplot as plt
    import numpy as np

    doses = sorted(eval_df["dose"].unique(), key=lambda d: (str(d) == "all", str(d)))
    matrix = np.full((len(CONSTRUCTIONS), len(doses)), np.nan)
    for i, cons in enumerate(CONSTRUCTIONS):
        for j, d in enumerate(doses):
            accs = eval_df[
                (eval_df["model_construction"] == cons) & (eval_df["dose"].astype(str) == str(d))
            ]["accuracy"].to_numpy(dtype=float)
            if accs.size >= 2 and accs.mean() != 0:
                matrix[i, j] = accs.std() / accs.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(doses)))
    ax.set_xticklabels([str(d) for d in doses])
    ax.set_yticks(range(len(CONSTRUCTIONS)))
    ax.set_yticklabels([DISPLAY_NAMES.get(c, c) for c in CONSTRUCTIONS])
    ax.set_xlabel("Dose")
    ax.set_ylabel("Construction")
    ax.set_title("Coefficient of variation across seeds")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CV (std / mean)")

    # Annotate cells so reviewers can read exact values.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def fig4_ngram_comparison(
    eval_df: pd.DataFrame, ngram_df: pd.DataFrame, out_path: Path
) -> None:
    """2x2 panel comparing the neural LM mean curve against the 4-gram baseline."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    fig.suptitle("Neural LM vs n-gram baseline")

    def mean_curve(df, cons):
        sub = df[df["model_construction"] == cons]
        doses = sorted(sub["dose_value"].unique())
        means = [sub.loc[sub["dose_value"] == d, "accuracy"].mean() for d in doses]
        return np.log10(np.asarray(doses, dtype=float) + 1.0), means

    for ax, cons in zip(axes.flat, CONSTRUCTIONS, strict=True):
        xl, ml = mean_curve(eval_df, cons)
        ax.plot(xl, ml, color=_color_for(cons), lw=2.0, marker="o", label="LTG-BERT")
        if not ngram_df.empty and cons in set(ngram_df["model_construction"]):
            xn, mn = mean_curve(ngram_df, cons)
            ax.plot(xn, mn, color="black", lw=1.5, ls="--", marker="s", label="4-gram")
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
        ax.set_title(DISPLAY_NAMES.get(cons, cons))
        ax.set_xlabel(r"$\log_{10}(\mathrm{dose} + 1)$")
        ax.set_ylabel("SLOR accuracy")
        ax.set_ylim(0.0, 1.0)
        _apply_minimal_theme(ax)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def fig5_collateral_effects(eval_df_full: pd.DataFrame, out_path: Path) -> None:
    """4x4 heatmap: train construction (rows) vs eval construction (cols), mean acc.

    The off-diagonal is the interesting part — does training on one construction
    drag a model's score on a different one? That's collateral learning, and it
    needs the full cross-eval table, not just the self-eval rows.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.full((len(CONSTRUCTIONS), len(CONSTRUCTIONS)), np.nan)
    for i, train in enumerate(CONSTRUCTIONS):
        for j, ev in enumerate(CONSTRUCTIONS):
            accs = eval_df_full[
                (eval_df_full["model_construction"] == train)
                & (eval_df_full["eval_construction"] == ev)
            ]["accuracy"].to_numpy(dtype=float)
            if accs.size:
                matrix[i, j] = accs.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CONSTRUCTIONS)))
    ax.set_xticklabels([DISPLAY_NAMES.get(c, c) for c in CONSTRUCTIONS],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(CONSTRUCTIONS)))
    ax.set_yticklabels([DISPLAY_NAMES.get(c, c) for c in CONSTRUCTIONS])
    ax.set_xlabel("Evaluated on")
    ax.set_ylabel("Trained on")
    ax.set_title("Collateral effects: mean accuracy across train/eval pairs")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean SLOR accuracy")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def run(config_path: Path) -> Path:
    """Generate all five figures into ``results/figures/``."""
    import pandas as pd

    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eval_csv = results_dir / "eval_results.csv"
    hill_csv = results_dir / "hill_fits.csv"
    ngram_csv = results_dir / "ngram_baseline.csv"
    sanity_path = results_dir / "dose_corpora_sanity.json"
    _require(eval_csv, "eval_results.csv")
    _require(hill_csv, "hill_fits.csv")

    eval_full = pd.read_csv(eval_csv)
    eval_full = resolve_dose_values(eval_full, sanity_path)
    eval_self = eval_full[eval_full["model_construction"] == eval_full["eval_construction"]].copy()
    hill_fits = pd.read_csv(hill_csv)

    fig1_dose_response_curves(eval_self, hill_fits, fig_dir / "fig1_dose_response_curves.pdf")
    fig2_parameter_clusters(hill_fits, fig_dir / "fig2_parameter_clusters.pdf")
    fig3_seed_variance(eval_self, fig_dir / "fig3_seed_variance.pdf")

    if ngram_csv.exists():
        ngram = resolve_dose_values(pd.read_csv(ngram_csv), sanity_path)
        ngram = ngram[ngram["model_construction"] == ngram["eval_construction"]].copy()
    else:
        logger.warning("No %s; fig4 will show the LM curve only.", ngram_csv)
        ngram = pd.DataFrame(columns=eval_self.columns)
    fig4_ngram_comparison(eval_self, ngram, fig_dir / "fig4_ngram_comparison.pdf")

    fig5_collateral_effects(eval_full, fig_dir / "fig5_collateral_effects.pdf")

    logger.info("All figures written to %s", fig_dir)
    return fig_dir


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
