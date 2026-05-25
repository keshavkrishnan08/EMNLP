"""Predict a construction's learnability from its corpus footprint — RQ4.

The headline target is now ``E0``, the indirect-evidence floor: how well a model
handles a construction it never saw directly. ``E0`` exists for *all eight*
constructions (it only needs the zero-dose model), where the threshold ``E50``
exists only for the four *core* constructions whose full dose ladder we fit. So
the main analysis predicts E0 across all eight, and E50 across the four core as
a smaller secondary check. Both are exploratory.

**Pre-registered predictor set.** These four predictors, their definitions, and
the choice of E0 as the headline target were fixed *before* looking at the
fitted values — they're a-priori and, crucially, non-circular: none of them is
the construction's own attested count dressed up. The headline predictors:

* **neighbor_density** — how much *structurally related but distinct* evidence
  surrounds the construction. NON-CIRCULAR by construction: it is built only
  from the OTHER constructions' positives, never this one's. We use a documented
  lexical-overlap proxy: for each other construction we weight its attested
  count by the Jaccard overlap between its content-token vocabulary and this
  construction's, then sum. High when sibling constructions are both frequent
  and lexically similar — a stand-in for "indirect evidence is nearby". (A POS
  n-gram skeleton would be cleaner but needs a Stanza pass; the lexical proxy is
  the cheap, documented approximation.)
* **productivity** — type/token ratio of content-word lemmas across the
  construction's own attested instances. A proxy for lexical openness: a
  construction that recurs with the same few words is less "productive" than one
  that hosts a fresh vocabulary each time. Lemmas are approximated with
  lowercased whitespace tokens minus a small stopword set (see ``_STOPWORDS``).
* **surface_predictability** — mean per-token unigram log-prob of the attested
  instances under a unigram model built from those same instances (n-gram
  surface recoverability). Higher means the construction lives in
  high-frequency, easy-to-anticipate words. Reuses
  ``drc.eval.slor.build_unigram_counts`` / ``unigram_logprob`` so the units
  match the SLOR pipeline exactly.

* **attested_count** — how many positive instances the corpus holds. Kept as a
  SEPARATE predictor, expected to be weak and explicitly NOT the headline; it's
  the raw-exposure baseline the others should beat. Read from the dose sanity
  JSON's ``dose-all`` entry when present, else the positives ``.jsonl`` line
  count.

Then we join with E0 (all eight) and E50 (four core) and correlate each
predictor against each — Spearman *and* Pearson, because with so few points a
single outlier can flip Pearson while Spearman holds.

A loud honesty caveat, baked into the output and the logs: **there are only
eight constructions, four of them core.** Seeds give more rows, but those are
pseudo-replicates — the predictors are identical within a construction. We
correlate at the construction level (n=8 for E0, n=4 for E50) and flag the whole
thing as exploratory, not an established law. Treat the p-values as decoration.

CLI::

    python -m drc.analysis.predictability --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drc.data.download import load_config, resolve_path
from drc.design import all_constructions

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pandas lazy
    import pandas as pd

logger = logging.getLogger(__name__)

# Targets we try to predict, with the construction set each is defined over.
# E0 (the indirect-evidence floor) is the headline and exists for ALL eight
# constructions; E50 exists only for the four core ones whose ladder we fit.
TARGET_SCOPES = {"E0": "all", "E50": "core"}
TARGETS = tuple(TARGET_SCOPES)

# Pre-registered predictor columns, in the order they're written and correlated.
# neighbor_density is the headline (non-circular, sibling-only); attested_count
# trails as the deliberately-weak raw-exposure baseline.
PREDICTORS = (
    "neighbor_density",
    "productivity",
    "surface_predictability",
    "attested_count",
)

# A deliberately tiny closed-class stopword list. The productivity ratio is
# meant to capture *content*-word openness, so we strip the function words that
# every instance shares regardless of the construction. Kept short on purpose:
# an aggressive list would be its own confound. Documented as an approximation.
_STOPWORDS = frozenset(
    """
    a an the this that these those
    is are was were be been being am
    do does did done has have had having
    of to in on at by for with from as into onto
    and or but nor so yet
    i you he she it we they me him her us them
    my your his its our their
    not no
    """.split()
)


def _positives_path(filtered_dir: Path, cons: str) -> Path:
    """Where a construction's attested-positives ``.jsonl`` lives."""
    return filtered_dir / f"{cons}_positives.jsonl"


def _read_positive_texts(path: Path) -> list[str]:
    """Pull one surface string per attested instance from a positives ``.jsonl``.

    The filter stage writes line-delimited JSON. We accept whichever text field
    is present — ``text``, ``sentence``, or ``good_sentence`` — and fall back to
    the raw line if a record is a bare string. Blank lines are skipped. A record
    with no usable text is dropped with a warning rather than crashing the run;
    one malformed line shouldn't sink the whole analysis.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Positives file not found at {path}. Run the filter/dose stage so "
            "data/filtered/<construction>_positives.jsonl exists."
        )

    texts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a plain-text positives dump too.
                texts.append(line)
                continue
            if isinstance(obj, str):
                texts.append(obj)
                continue
            text = (
                obj.get("text")
                or obj.get("sentence")
                or obj.get("good_sentence")
            )
            if text and str(text).strip():
                texts.append(str(text).strip())
            else:
                logger.warning("%s:%d has no text field; skipping.", path.name, lineno)
    return texts


def _content_tokens(text: str) -> list[str]:
    """Lowercased whitespace tokens minus stopwords — the lemma stand-in.

    Whitespace tokenisation is crude, but the productivity ratio only needs a
    consistent rule applied to every construction, not linguistic truth. We
    strip leading/trailing punctuation so ``"dog,"`` and ``"dog"`` count once.
    """
    out: list[str] = []
    for tok in text.lower().split():
        word = tok.strip(".,!?;:\"'()[]{}—–-")
        if word and word not in _STOPWORDS:
            out.append(word)
    return out


def _attested_count_from_sanity(sanity: dict[str, Any], cons: str) -> int | None:
    """Read the construction's corpus-wide positive count from the sanity JSON.

    ``dose_corpora_sanity.json`` keys look like ``"<cons>_dose-all"`` mapping to
    ``{"observed": int, ...}``. The ``dose-all`` row is the total we want. We
    accept a couple of shapes so a hand-edited file still works, and return None
    when there's nothing usable so the caller can fall back to the line count.
    """
    for key in (f"{cons}_dose-all", f"{cons}_dose-All"):
        payload = sanity.get(key)
        if isinstance(payload, dict):
            val = payload.get("observed", payload.get("attested_all"))
            if val is not None:
                return int(val)
        elif payload is not None:
            return int(payload)
    # Some files key by bare construction name instead.
    payload = sanity.get(cons)
    if isinstance(payload, dict):
        val = payload.get("attested_all") or payload.get("n_all")
        if val is not None:
            return int(val)
    elif payload is not None:
        return int(payload)
    return None


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard overlap of two token-vocabulary sets; 0 when both empty."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _neighbor_density(
    cons: str,
    vocab: dict[str, set[str]],
    counts: dict[str, int],
) -> float:
    """Structurally-related-but-distinct evidence around ``cons``. NON-CIRCULAR.

    Built only from the OTHER constructions: sum over each other construction of
    (its attested count) x (Jaccard overlap of its content vocabulary with this
    construction's). The target construction's own count never enters, so this
    can't collapse into ``attested_count``. High when sibling constructions are
    both frequent and lexically similar — the documented proxy for "indirect
    evidence is nearby".
    """
    me = vocab.get(cons, set())
    total = 0.0
    for other, vec in vocab.items():
        if other == cons:
            continue
        total += counts.get(other, 0) * _jaccard(me, vec)
    return float(total)


def compute_predictors(
    filtered_dir: Path,
    sanity_path: Path | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build the per-construction predictor table from the attested positives.

    One row per construction (all eight), with the pre-registered predictor
    columns. ``productivity`` and ``surface_predictability`` come from the
    construction's own positives; ``neighbor_density`` is built only from the
    OTHER constructions (non-circular); ``attested_count`` is the raw-exposure
    baseline. Raises if a positives file is missing — better a clear stop than a
    silently short table.
    """
    import pandas as pd

    from drc.eval.slor import build_unigram_counts, unigram_logprob

    sanity: dict[str, Any] = {}
    if sanity_path is not None and sanity_path.exists():
        with open(sanity_path, encoding="utf-8") as fh:
            sanity = json.load(fh)
    else:
        logger.warning(
            "No dose sanity JSON at %s; attested_count falls back to line counts.",
            sanity_path,
        )

    constructions = all_constructions(config)

    # First pass: read positives, compute per-construction own-vocabulary and
    # attested counts. neighbor_density needs every construction's vocab, so it's
    # computed in a second pass once they're all in hand.
    texts_by: dict[str, list[str]] = {}
    vocab_by: dict[str, set[str]] = {}
    count_by: dict[str, int] = {}
    for cons in constructions:
        texts = _read_positive_texts(_positives_path(filtered_dir, cons))
        if not texts:
            raise RuntimeError(
                f"No attested instances read for '{cons}'. The positives file is "
                "empty; can't compute predictors."
            )
        texts_by[cons] = texts
        toks: list[str] = []
        for t in texts:
            toks.extend(_content_tokens(t))
        vocab_by[cons] = set(toks)
        n_attested = _attested_count_from_sanity(sanity, cons)
        count_by[cons] = int(n_attested) if n_attested is not None else len(texts)

    rows: list[dict[str, Any]] = []
    for cons in constructions:
        texts = texts_by[cons]
        n_attested = count_by[cons]

        # productivity: type/token ratio over content lemmas, pooled across all
        # instances. Ranges (0, 1]; near 1 means almost no lexical repetition.
        all_tokens: list[str] = []
        for t in texts:
            all_tokens.extend(_content_tokens(t))
        n_tokens = len(all_tokens)
        productivity = (len(set(all_tokens)) / n_tokens) if n_tokens else float("nan")

        # surface_predictability: mean per-token unigram log-prob, where the
        # unigram model is the positives themselves. We borrow the SLOR helpers
        # via a temp file so the tokenisation matches the eval pipeline exactly.
        counts = _unigram_from_texts(texts, build_unigram_counts)
        per_sent = []
        for t in texts:
            n_words = max(1, len(_content_tokens(t)))
            per_sent.append(unigram_logprob(t, counts) / n_words)
        surface_predictability = float(sum(per_sent) / len(per_sent))

        # neighbor_density: sibling-only, non-circular (see _neighbor_density).
        neighbor_density = _neighbor_density(cons, vocab_by, count_by)

        rows.append(
            {
                "construction": cons,
                "neighbor_density": neighbor_density,
                "productivity": float(productivity),
                "surface_predictability": surface_predictability,
                "attested_count": int(n_attested),
                "n_instances_read": len(texts),
            }
        )
        logger.info(
            "%s: neighbor=%.2f productivity=%.3f surface=%.3f count=%d (from %d)",
            cons, neighbor_density, productivity, surface_predictability,
            n_attested, len(texts),
        )

    return pd.DataFrame(rows)


def _unigram_from_texts(texts: list[str], build_unigram_counts) -> Counter[str]:
    """Build SLOR unigram counts from in-memory texts.

    ``build_unigram_counts`` reads a file, so we hand it a throwaway temp file
    of one sentence per line. Reusing it (rather than re-counting here) keeps the
    tokenisation byte-identical to what SLOR uses downstream.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        for t in texts:
            fh.write(t.replace("\n", " ") + "\n")
        tmp = Path(fh.name)
    try:
        return build_unigram_counts(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _construction_level_targets(hill_fits: pd.DataFrame) -> pd.DataFrame:
    """Mean (and std) of each target per construction, over seeds.

    We collapse seeds to one row per construction because the predictors are
    constant within a construction — correlating at the seed level would just
    inflate n with copies. The std comes along so the figure can draw error bars.

    Each target is aggregated over rows where it is finite. That matters for the
    tiered design: ``E0`` exists for all eight constructions (breadth rows carry
    it even though their Hill fit didn't converge), while ``E50`` is NaN for
    breadth and so collapses to the four core constructions on its own. We do
    NOT pre-filter to ``converged`` here — that would throw away breadth E0.
    """
    import numpy as np

    if hill_fits.empty:
        raise RuntimeError("No Hill-fit rows to correlate against.")

    frames: list[pd.DataFrame] = []
    for t in TARGETS:
        if t not in hill_fits:
            continue
        sub = hill_fits[np.isfinite(hill_fits[t].astype(float))]
        if sub.empty:
            continue
        agg = sub.groupby("construction")[t].agg(["mean", "std"]).reset_index()
        agg = agg.rename(columns={"mean": f"{t}_mean", "std": f"{t}_std"})
        frames.append(agg)

    if not frames:
        raise RuntimeError("No finite target values to correlate against.")

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="construction", how="outer")
    return out


def correlate(
    predictors: pd.DataFrame, hill_fits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join predictors to per-construction Hill targets and correlate them.

    Returns ``(merged, correlations)``. ``merged`` is the construction-level
    table (predictors + mean/std targets). ``correlations`` has one row per
    (predictor, target) pair with Spearman and Pearson coefficients, their
    p-values, and ``n_constructions`` — the number of constructions the target
    is defined over (eight for E0, four for E50), meant to be read as a warning
    label. Every row is flagged ``exploratory_only``.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    targets = _construction_level_targets(hill_fits)
    merged = predictors.merge(targets, on="construction", how="inner")
    if merged.empty:
        raise RuntimeError("No constructions matched between predictors and fits.")

    rows: list[dict[str, Any]] = []
    for pred in PREDICTORS:
        x = merged[pred].to_numpy(dtype=float)
        for t in TARGETS:
            col = f"{t}_mean"
            if col not in merged:
                continue
            y = merged[col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
                sr = sp = pr = pp = float("nan")
            else:
                sr, sp = stats.spearmanr(x[mask], y[mask])
                pr, pp = stats.pearsonr(x[mask], y[mask])
            rows.append(
                {
                    "predictor": pred,
                    "target": t,
                    "target_scope": TARGET_SCOPES.get(t, "all"),
                    "n_constructions": int(mask.sum()),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "exploratory_only": True,
                }
            )
    correlations = pd.DataFrame(rows)
    return merged, correlations


def _strongest_predictor(correlations: pd.DataFrame, target: str = "E0") -> str:
    """The predictor with the largest |Spearman r| against ``target``.

    Spearman because it's the rank measure that survives a tiny sample best.
    Ties and all-NaN rows fall back to the first predictor so the figure always
    has something to draw.
    """
    sub = correlations[correlations["target"] == target].copy()
    sub["abs_r"] = sub["spearman_r"].abs()
    sub = sub.sort_values("abs_r", ascending=False, na_position="last")
    if sub.empty or not sub["abs_r"].notna().any():
        return PREDICTORS[0]
    return str(sub.iloc[0]["predictor"])


def fig6_predictability(
    merged: pd.DataFrame,
    correlations: pd.DataFrame,
    out_path: Path,
    target: str = "E0",
) -> None:
    """Scatter ``target`` against its strongest predictor, one point per construction.

    Defaults to the headline target ``E0`` (across all eight constructions).
    Points are construction means; vertical error bars are the seed std. We
    label each point and drop the Spearman r in the corner so the figure carries
    its own honesty caveat. Okabe-Ito colours, matching the rest of the figures.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Local import to avoid a hard figures.py dependency at module import time.
    from drc.analysis.figures import (
        DISPLAY_NAMES,
        OKABE_ITO,
        _apply_minimal_theme,
    )

    pred = _strongest_predictor(correlations, target=target)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    plotted = merged[np.isfinite(merged[f"{target}_mean"].astype(float))]
    for i, (_, row) in enumerate(plotted.iterrows()):
        cons = row["construction"]
        color = OKABE_ITO[i % len(OKABE_ITO)]
        x = float(row[pred])
        y = float(row[f"{target}_mean"])
        yerr = float(row.get(f"{target}_std", float("nan")))
        yerr = 0.0 if not np.isfinite(yerr) else yerr
        ax.errorbar(
            x, y, yerr=yerr, fmt="o", markersize=10,
            color=color, ecolor=color,
            elinewidth=1.2, capsize=4, markeredgecolor="black",
            markeredgewidth=0.5,
        )
        ax.annotate(
            DISPLAY_NAMES.get(cons, cons), (x, y),
            textcoords="offset points", xytext=(8, 4), fontsize=8,
        )

    corr_row = correlations[
        (correlations["predictor"] == pred) & (correlations["target"] == target)
    ]
    n_pts = int(len(plotted))
    if not corr_row.empty:
        sr = corr_row.iloc[0]["spearman_r"]
        pr = corr_row.iloc[0]["pearson_r"]
        ax.text(
            0.02, 0.98,
            f"Spearman r = {sr:.2f}\nPearson r = {pr:.2f}\n(n = {n_pts}, exploratory)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "ec": "grey", "alpha": 0.8},
        )

    pretty = pred.replace("_", " ")
    ax.set_xlabel(f"{pretty} (corpus predictor)")
    ylabel = r"$E_0$ (mean over seeds)" if target == "E0" else (
        r"$E_{50}$ (mean over seeds)" if target == "E50" else f"{target} (mean over seeds)"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"Predicting {target} from corpus properties (exploratory)")
    _apply_minimal_theme(ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s (predictor=%s)", out_path, pred)


def run(config_path: Path) -> dict[str, Path]:
    """Compute predictors, correlate against the Hill fits, write CSVs + figure."""
    import pandas as pd

    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    filtered_dir = resolve_path(config_path, config["paths"]["filtered"])
    sanity_path = results_dir / "dose_corpora_sanity.json"
    hill_csv = results_dir / "hill_fits.csv"

    if not hill_csv.exists():
        raise FileNotFoundError(
            f"Hill fits not found at {hill_csv}. Run `python -m drc.analysis.hill` first."
        )

    predictors = compute_predictors(filtered_dir, sanity_path, config)
    hill_fits = pd.read_csv(hill_csv)
    merged, correlations = correlate(predictors, hill_fits)

    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / "predictability.csv"
    corr_path = results_dir / "predictability_correlations.csv"
    merged.to_csv(pred_path, index=False)
    correlations.to_csv(corr_path, index=False)
    logger.info("Wrote %s and %s", pred_path, corr_path)

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "fig6_predictability.pdf"
    fig6_predictability(merged, correlations, fig_path)

    return {"predictors": pred_path, "correlations": corr_path, "figure": fig_path}


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
