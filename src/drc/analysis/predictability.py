"""Predict a construction's learnability from its corpus footprint — RQ4.

The Hill fits hand us three readable numbers per construction: a floor ``E0``,
a half-learning dose ``E50``, and a sharpness ``n``. The obvious next question
is whether you could have *guessed* those numbers ahead of training, from cheap
properties of how the construction shows up in the corpus. If E50 tracks, say,
how lexically open a construction is, that's a learnability law in miniature —
and as far as we can tell nobody's tried it. Prior work (WIDET/Oba 2024, the
AANN line) measures exposure effects but never extracts a threshold, let alone
predicts one.

So we compute three predictors per construction, straight from the attested
positives and corpus counts:

* **attested_count** — how many positive instances the corpus holds. The raw
  amount of evidence. Read from the dose sanity JSON's ``dose-all`` entry when
  it's there, else the positives ``.jsonl`` line count.
* **productivity** — type/token ratio of content-word lemmas across the
  attested instances. A proxy for lexical openness: a construction that recurs
  with the same few words is less "productive" than one that hosts a fresh
  vocabulary each time. We approximate lemmas with lowercased whitespace tokens
  minus a small stopword set (see ``_STOPWORDS``); a true lemmatiser would be
  better but isn't worth a Stanza dependency for a four-point exploratory fit.
* **surface_predictability** — mean per-token unigram log-prob of the attested
  instances under a unigram model built from those same instances. Higher means
  the construction lives in high-frequency, easy-to-anticipate words. We reuse
  ``drc.eval.slor.build_unigram_counts`` / ``unigram_logprob`` so the units
  match the SLOR pipeline exactly.

Then we join with the Hill fits and correlate each predictor against E50, E0,
and n — Spearman *and* Pearson, because with four points a single outlier can
flip Pearson while Spearman holds.

A loud honesty caveat, baked into the output and the logs: **there are only
four constructions.** Three seeds each gives twelve rows, but those are
pseudo-replicates, not independent constructions — the predictors are identical
within a construction. We report correlations at the construction level (n=4)
and flag the whole thing as a proof-of-concept, not an established law. Treat
the p-values as decoration.

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

from drc import CONSTRUCTIONS
from drc.data.download import load_config, resolve_path

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pandas lazy
    import pandas as pd

logger = logging.getLogger(__name__)

# The Hill parameters we try to predict. E50 is the headline (how much data the
# construction needs); E0 and n come along because they're cheap to report.
TARGETS = ("E50", "E0", "n")

# Predictor columns, in the order they're written and correlated.
PREDICTORS = ("attested_count", "productivity", "surface_predictability")

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


def compute_predictors(
    filtered_dir: Path, sanity_path: Path | None
) -> pd.DataFrame:
    """Build the per-construction predictor table from the attested positives.

    One row per construction, columns ``attested_count``, ``productivity``,
    ``surface_predictability``. Each predictor is computed from that
    construction's own positives so the three numbers describe the same set of
    sentences. Raises if a positives file is missing — better a clear stop than
    a silently short table.
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

    rows: list[dict[str, Any]] = []
    for cons in CONSTRUCTIONS:
        texts = _read_positive_texts(_positives_path(filtered_dir, cons))
        if not texts:
            raise RuntimeError(
                f"No attested instances read for '{cons}'. The positives file is "
                "empty; can't compute predictors."
            )

        # attested_count: prefer the audited corpus total, else how many we read.
        n_attested = _attested_count_from_sanity(sanity, cons)
        if n_attested is None:
            n_attested = len(texts)

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

        rows.append(
            {
                "construction": cons,
                "attested_count": int(n_attested),
                "productivity": float(productivity),
                "surface_predictability": surface_predictability,
                "n_instances_read": len(texts),
            }
        )
        logger.info(
            "%s: count=%d productivity=%.3f surface=%.3f (from %d instances)",
            cons, n_attested, productivity, surface_predictability, len(texts),
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
    """Mean (and std) of each Hill target per construction, over seeds.

    We collapse seeds to one row per construction because the predictors are
    constant within a construction — correlating at the seed level would just
    inflate n with copies. The std comes along so the figure can draw error bars.
    """
    good = hill_fits[hill_fits["converged"].astype(bool)] if "converged" in hill_fits else hill_fits
    if good.empty:
        raise RuntimeError("No converged Hill fits to correlate against.")

    agg: dict[str, Any] = {}
    for t in TARGETS:
        agg[f"{t}_mean"] = (t, "mean")
        agg[f"{t}_std"] = (t, "std")
    out = good.groupby("construction").agg(**agg).reset_index()
    return out


def correlate(
    predictors: pd.DataFrame, hill_fits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join predictors to per-construction Hill targets and correlate them.

    Returns ``(merged, correlations)``. ``merged`` is the construction-level
    table (predictors + mean/std targets). ``correlations`` has one row per
    (predictor, target) pair with Spearman and Pearson coefficients, their
    p-values, and the sample size ``n`` — which here is the number of
    constructions, four, and is meant to be read as a warning label.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    targets = _construction_level_targets(hill_fits)
    merged = predictors.merge(targets, on="construction", how="inner")
    n = len(merged)
    if n < 3:
        raise RuntimeError(
            f"Only {n} constructions after the join; need >= 3 for a correlation."
        )

    rows: list[dict[str, Any]] = []
    for pred in PREDICTORS:
        x = merged[pred].to_numpy(dtype=float)
        for t in TARGETS:
            y = merged[f"{t}_mean"].to_numpy(dtype=float)
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


def _strongest_predictor(correlations: pd.DataFrame, target: str = "E50") -> str:
    """The predictor with the largest |Spearman r| against ``target``.

    Spearman because it's the rank measure that survives a four-point fit best.
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
    target: str = "E50",
) -> None:
    """Scatter E50 against its strongest predictor, one point per construction.

    Points are construction means; vertical error bars are the seed std of the
    target. We label each point and drop the Spearman r in the corner so the
    figure carries its own honesty caveat. Okabe-Ito colours, one per
    construction, matching the rest of the figure set.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Local import to avoid a hard figures.py dependency at module import time.
    from drc.analysis.figures import (
        DISPLAY_NAMES,
        _apply_minimal_theme,
        _color_for,
    )

    pred = _strongest_predictor(correlations, target=target)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for _, row in merged.iterrows():
        cons = row["construction"]
        x = float(row[pred])
        y = float(row[f"{target}_mean"])
        yerr = float(row.get(f"{target}_std", float("nan")))
        yerr = 0.0 if not np.isfinite(yerr) else yerr
        ax.errorbar(
            x, y, yerr=yerr, fmt="o", markersize=10,
            color=_color_for(cons), ecolor=_color_for(cons),
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
    if not corr_row.empty:
        sr = corr_row.iloc[0]["spearman_r"]
        pr = corr_row.iloc[0]["pearson_r"]
        ax.text(
            0.02, 0.98,
            f"Spearman r = {sr:.2f}\nPearson r = {pr:.2f}\n(n = 4, exploratory)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "ec": "grey", "alpha": 0.8},
        )

    pretty = pred.replace("_", " ")
    ax.set_xlabel(f"{pretty} (corpus predictor)")
    ax.set_ylabel(r"$E_{50}$ (mean over seeds)" if target == "E50" else f"{target} (mean over seeds)")
    ax.set_title(f"Predicting {target} from corpus properties (proof of concept)")
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

    predictors = compute_predictors(filtered_dir, sanity_path)
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
