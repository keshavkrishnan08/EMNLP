"""The indirect-evidence index --- the paper's headline number.

Here's the question this module answers. If a model never sees a construction
directly during pretraining, how well does it handle that construction anyway?
Whatever accuracy it scores at *zero* exposure can only have come from indirect
evidence: related patterns, shared sub-structure, distributional echoes of the
target. We call that floor ``E0``, and reading it across a wide set of
constructions is the whole point of the tiered design.

For every construction we compute three things straight from the eval table:

* **E0** --- zero-exposure accuracy. Mean over seeds, with a standard error so
  the spread is visible. This is the indirect-evidence floor.
* **Emax** --- the shared-ceiling accuracy on that construction, read from the
  one full-corpus model (``model_construction == "full"``, ``dose == "all"``)
  evaluated on the construction's own minimal pairs. See
  ``drc.design.emax_construction``.
* **direct_value** = ``Emax - E0`` --- how much *direct* exposure buys you on
  top of what indirect evidence already gave. A small direct value means the
  construction was nearly free; a large one means exposure mattered.

**The E0 contamination bound.** A dose-0 corpus is supposed to contain zero
instances of the construction, but the filter that strips them isn't perfect.
If the filter's recall is ``r``, it misses a fraction ``(1 - r)`` of the true
instances, and those leak back into the dose-0 corpus. So E0 might be inflated
by a little real exposure. We do NOT correct E0 for this --- we report it as an
upper bound, a caveat column, so nobody can claim E0 is silently padded. The
leaked fraction relative to what was caught is ``(1 - r) / r``; we surface that
and the recall we used, flagging whether the recall came from a real audit or a
default. The math is deliberately simple: it's a bound, not an estimate.

Filter recall comes from ``data/qa_audits/<cons>_audit.csv`` when a labeled
audit exists (scored via ``drc.data.qa_audit.score_audit``). When it doesn't,
we fall back to a configurable default recall and flag the row clearly.

CLI::

    python -m drc.analysis.indirect_evidence --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drc.data.download import load_config, resolve_path
from drc.design import all_constructions, emax_construction

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pandas lazy
    import pandas as pd

logger = logging.getLogger(__name__)

# Recall we assume when no labeled audit exists. The PRD acceptance floor is
# recall >= 0.85 (see drc.data.filters.base.FilterReport.passes), so 0.85 is the
# conservative worst case that still passes the gate --- it maximises the
# leakage bound rather than flattering E0. Override via --default-recall.
DEFAULT_FILTER_RECALL = 0.85


def _audit_csv_path(audits_dir: Path, cons: str) -> Path:
    """Where a construction's labeled QA audit CSV lives (qa_audit layout)."""
    return audits_dir / f"{cons}_audit.csv"


def filter_recall(audits_dir: Path, cons: str, default: float) -> tuple[float, str]:
    """Read filter recall for a construction, else fall back to a default.

    Returns ``(recall, source)`` where ``source`` is ``"audit"`` when the value
    came from a scored ``data/qa_audits/<cons>_audit.csv`` and ``"default"``
    otherwise. We never invent an audit: a missing or unlabeled CSV falls back
    cleanly with a flagged source so the caveat column stays honest.
    """
    path = _audit_csv_path(audits_dir, cons)
    if path.exists():
        try:
            from drc.data.qa_audit import score_audit

            report = score_audit(path, cons)
            return float(report.recall), "audit"
        except Exception as exc:  # noqa: BLE001 - degrade to default, stay loud
            logger.warning(
                "Could not score audit %s (%s); using default recall %.2f.",
                path, exc, default,
            )
    else:
        logger.info(
            "No audit CSV for %s at %s; using default recall %.2f.",
            cons, path, default,
        )
    return float(default), "default"


def _leakage_bound_note(recall: float, source: str) -> str:
    """A short, self-documenting caveat string for the leakage upper bound.

    The leaked fraction relative to caught instances is ``(1 - r) / r``. We
    state it as a percentage upper bound on dose-0 contamination, not a
    correction, and name the recall source so a reader knows if it's audited.
    """
    if recall <= 0:
        return f"recall={recall:.2f} ({source}); leakage bound undefined (recall<=0)"
    leaked_frac = (1.0 - recall) / recall
    return (
        f"recall={recall:.2f} ({source}); up to ~{leaked_frac * 100:.1f}% of caught "
        "instances may have leaked into dose-0 (UPPER BOUND, E0 not corrected)"
    )


def compute_indirect_evidence(
    eval_csv: Path,
    config: dict[str, Any],
    audits_dir: Path | None = None,
    default_recall: float = DEFAULT_FILTER_RECALL,
) -> pd.DataFrame:
    """Per-construction E0, Emax, direct_value and the E0 contamination bound.

    E0 is the mean over seeds of dose-0 self-eval accuracy, with a standard
    error (sample std / sqrt(n_seeds)). Emax is the shared-ceiling model's
    accuracy on the construction. Raises if the eval table or the required rows
    are missing for a construction --- a silently short table would understate
    the index.
    """
    import numpy as np
    import pandas as pd

    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Eval results not found at {eval_csv}. Run the eval stage first."
        )

    df = pd.read_csv(eval_csv)
    dose_str = df["dose"].astype(str).str.strip().str.lower()

    rows: list[dict[str, Any]] = []
    for cons in all_constructions(config):
        on_cons = df[df["eval_construction"] == cons]

        # E0: dose-0 rows of the construction's own zero-dose model, over seeds.
        zero = on_cons[
            (on_cons["model_construction"] == cons)
            & (dose_str.loc[on_cons.index] == "0")
        ]
        if zero.empty:
            raise RuntimeError(
                f"No dose-0 eval rows for construction '{cons}'. E0 needs the "
                "zero-dose model evaluated on its own minimal pairs."
            )
        e0_vals = zero["accuracy"].to_numpy(dtype=float)
        e0 = float(np.mean(e0_vals))
        e0_se = (
            float(np.std(e0_vals, ddof=1) / np.sqrt(e0_vals.size))
            if e0_vals.size > 1
            else 0.0
        )

        # Emax: the shared ceiling model evaluated on this construction.
        ceiling_code = emax_construction(config, cons)
        ceiling = on_cons[
            (on_cons["model_construction"] == ceiling_code)
            & (dose_str.loc[on_cons.index] == "all")
        ]
        if ceiling.empty:
            raise RuntimeError(
                f"No ceiling row for '{cons}' (model_construction=="
                f"'{ceiling_code}', dose='all'). The shared full-corpus model "
                "supplies Emax for every construction."
            )
        emax = float(np.mean(ceiling["accuracy"].to_numpy(dtype=float)))

        recall, source = filter_recall(
            audits_dir, cons, default_recall
        ) if audits_dir is not None else (float(default_recall), "default")

        rows.append(
            {
                "construction": cons,
                "E0": e0,
                "E0_se": e0_se,
                "Emax": emax,
                "direct_value": emax - e0,
                "filter_recall": recall,
                "leakage_bound_note": _leakage_bound_note(recall, source),
            }
        )
        logger.info(
            "%s: E0=%.3f (+-%.3f) Emax=%.3f direct=%.3f recall=%.2f (%s)",
            cons, e0, e0_se, emax, emax - e0, recall, source,
        )

    return pd.DataFrame(rows)


def fig9_indirect_evidence(table: pd.DataFrame, out_path: Path) -> None:
    """Per-construction bar of E0 with the Emax ceiling marked, sorted by E0.

    Bars are E0 (with SE whiskers); a short horizontal tick marks Emax above
    each bar so the direct-exposure gap is visible at a glance. Okabe-Ito
    colours and the spare publication theme, matching the rest of figures.py.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from drc.analysis.figures import (
        DISPLAY_NAMES,
        OKABE_ITO,
        _apply_minimal_theme,
    )

    ordered = table.sort_values("E0", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered))
    e0 = ordered["E0"].to_numpy(dtype=float)
    e0_se = ordered["E0_se"].to_numpy(dtype=float)
    emax = ordered["Emax"].to_numpy(dtype=float)
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(ordered))]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(ordered)), 5.0))
    ax.bar(
        x, e0, yerr=e0_se, color=colors, edgecolor="black", linewidth=0.5,
        capsize=4, label=r"$E_0$ (indirect-evidence floor)",
    )
    # Emax ceiling marks: a horizontal tick spanning each bar.
    half = 0.4
    for xi, em in zip(x, emax, strict=True):
        ax.hlines(em, xi - half, xi + half, color="black", linewidth=1.5)
    ax.plot([], [], color="black", linewidth=1.5, label=r"$E_{\max}$ (ceiling)")

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)  # chance
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(c, c) for c in ordered["construction"]],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel("Minimal-pair accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Indirect-evidence floor ($E_0$) vs. ceiling ($E_{\\max}$)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    _apply_minimal_theme(ax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def run(config_path: Path, default_recall: float = DEFAULT_FILTER_RECALL) -> dict[str, Path]:
    """Compute the index, write ``results/indirect_evidence.csv`` + fig9."""
    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    eval_csv = results_dir / "eval_results.csv"
    audits_dir = resolve_path(config_path, "data/qa_audits")

    table = compute_indirect_evidence(
        eval_csv, config, audits_dir=audits_dir, default_recall=default_recall
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "indirect_evidence.csv"
    table.to_csv(csv_path, index=False)
    logger.info("Wrote %d constructions to %s", len(table), csv_path)

    fig_dir = results_dir / "figures"
    fig_path = fig_dir / "fig9_indirect_evidence.pdf"
    fig9_indirect_evidence(table, fig_path)

    return {"table": csv_path, "figure": fig_path}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--default-recall", type=float, default=DEFAULT_FILTER_RECALL,
        help="Filter recall to assume when no labeled QA audit is present.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, default_recall=args.default_recall)


if __name__ == "__main__":
    main()
