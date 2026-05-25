"""Memorization vs. generalization: does the construction curve rise on *novel* fillers?

This is the quantitative read on H3 — the paper's claim that grammar can come by
two routes, one that memorises the lexical material it saw and one that
generalises the abstract pattern. We never train anything new here. We reuse the
already-scored minimal pairs and split each construction's *own* test items into
two buckets:

* **seen** — the item's grammatical sentence shares at least one content word
  with some construction instance that survived into this dose corpus.
* **novel** — it shares nothing. The model never saw these particular words
  *inside the construction* at this dose.

Then we fit/plot the dose-response curve separately for each bucket. The logic is
simple. If the novel-filler curve still climbs with dose, the model is
generalising the construction past the exact items it was fed. If only the seen
curve climbs, it's leaning on memorised lexical material. WIDET / Oba (2024) dosed
constructions but never made this cut, so the split is what's new.

A caveat we'd rather state than bury: the overlap test is a *coarse proxy*. We
compare lowercased content-token sets and call an item "seen" on a single shared
content word. That over-counts overlap (a shared "dog" between two otherwise
unrelated sentences still flags "seen"), so the seen/novel gap here is a lower
bound on the real memorization signal, not a precise measurement. Dose 0 keeps no
instances, so every item is novel by construction — a useful sanity anchor.

CLI::

    python -m drc.analysis.generalization --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from drc import CONSTRUCTIONS
from drc.data.download import load_config, resolve_path

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pandas/numpy lazy
    import pandas as pd

logger = logging.getLogger(__name__)

# Columns in results/generalization.csv. One constant so the writer and any
# reader can't drift.
CSV_FIELDS = ("construction", "dose", "group", "n_items", "accuracy", "std_error")

# Group labels, used as literal strings in the CSV and the figure legend.
GROUP_SEEN = "seen"
GROUP_NOVEL = "novel"

# Same word shape the SLOR unigram side uses: lowercased word-ish runs. Keeping
# it identical means "content token" means the same thing on both sides of the
# pipeline.
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# A short, deliberately boring stopword list. The point is to drop function words
# so overlap is judged on *content* — nouns, verbs, adjectives — not on the "the"
# and "is" that every English sentence shares. We keep it small and explicit
# rather than pulling in a tagger; the filters already establish the construction,
# so we only need a rough content/function cut here.
_STOPWORDS = frozenset(
    """
    a an the this that these those
    is am are was were be been being
    do does did done
    have has had having
    will would shall should can could may might must
    of to in on at by for with from into onto off out up down over under
    and or but nor so yet for
    not no
    i you he she it we they me him her us them
    my your his its our their
    as if than then there here
    """.split()
)


def content_tokens(sentence: str) -> set[str]:
    """Lowercased content tokens of a sentence: word-ish runs minus stopwords.

    This is the unit of comparison for the overlap test. It's intentionally
    crude — no lemmatiser, no POS tagger — because the dose corpus already
    pinned down which sentences carry the construction. All we need here is a
    fast, deterministic content/function split.
    """
    return {w for w in _WORD_RE.findall(sentence.lower()) if w not in _STOPWORDS}


def kept_instance_vocab(conllu_path: Path) -> set[str]:
    """Pooled content vocabulary of every instance kept in a dose corpus.

    We read the ``# text =`` headers via the parser's metadata reader — the same
    text the model actually trained on — and union their content tokens. An
    empty corpus (or a missing file the caller has already vetted) yields an
    empty set, which makes every item "novel". That's the correct behaviour for
    dose 0.
    """
    from drc.data.parse import read_metadata

    vocab: set[str] = set()
    for sent in read_metadata(conllu_path):
        vocab |= content_tokens(sent.text)
    return vocab


def _dose_corpus_path(
    config: dict, config_path: Path, construction: str, dose: str
) -> Path:
    """Where the dose corpus for this run lives (matches dose_corpora.py naming)."""
    root = resolve_path(config_path, config["paths"]["dose_corpora"])
    return root / f"{construction}_dose-{dose}.conllu"


def _binomial_std_error(accuracy: float, n: int) -> float:
    """Standard error of a proportion: sqrt(p(1-p)/n). Zero when n is zero."""
    import math

    if n <= 0:
        return 0.0
    return math.sqrt(accuracy * (1.0 - accuracy) / n)


def classify_items(
    per_item: pd.DataFrame, kept_vocab: set[str]
) -> pd.DataFrame:
    """Tag each per-item row as seen/novel by content-word overlap with the corpus.

    Expects ``per_item`` already narrowed to one (construction, dose, seed) cell
    and to self-evaluation rows (eval == model construction). Needs a column the
    caller supplies — ``good_sentence`` — to read the lexical material from; the
    raw per-item CSV doesn't carry sentences, so the runner joins them back in
    from the eval_items files before calling here. Returns a copy with a new
    ``group`` column.

    An item is "seen" when its good sentence shares >= 1 content token with the
    pooled kept-instance vocabulary, else "novel".
    """
    out = per_item.copy()
    groups = []
    for sentence in out["good_sentence"]:
        toks = content_tokens(str(sentence))
        groups.append(GROUP_SEEN if (toks & kept_vocab) else GROUP_NOVEL)
    out["group"] = groups
    return out


def aggregate(labelled: pd.DataFrame) -> pd.DataFrame:
    """Collapse labelled per-item rows to per-(construction, dose, group) accuracy.

    Accuracy is the mean of ``correct`` over every item in the group, pooled
    across seeds — each seed contributes its items, so a group's n_items is the
    seed-summed item count. The standard error is the binomial proportion error
    over that pooled n. We keep both seen and novel groups for every dose even
    when one is empty, emitting an explicit zero-n row, so the figure has a
    stable grid to draw on.

    Returns a frame with exactly CSV_FIELDS, sorted for a deterministic file.
    """
    import numpy as np
    import pandas as pd

    rows: list[dict] = []
    for cons in sorted(labelled["model_construction"].unique()):
        cons_df = labelled[labelled["model_construction"] == cons]
        for dose in sorted(
            cons_df["dose"].unique(),
            key=lambda d: (str(d) == "all", str(d)),
        ):
            cell = cons_df[cons_df["dose"] == dose]
            for group in (GROUP_SEEN, GROUP_NOVEL):
                grp = cell[cell["group"] == group]
                n = int(len(grp))
                acc = float(np.mean(grp["correct"])) if n else 0.0
                rows.append(
                    {
                        "construction": cons,
                        "dose": dose,
                        "group": group,
                        "n_items": n,
                        "accuracy": round(acc, 6),
                        "std_error": round(_binomial_std_error(acc, n), 6),
                    }
                )
    return pd.DataFrame(rows, columns=list(CSV_FIELDS))


def build_generalization_table(
    per_item_csv: Path,
    eval_items_root: Path,
    dose_corpus_for: Callable[[str, str], Path | None],
) -> pd.DataFrame:
    """Core, I/O-light driver: read per-item scores, split, aggregate.

    Split out from :func:`run` so a test can call it on synthetic inputs without
    a config. ``dose_corpus_for(construction, dose)`` is a callback returning the
    .conllu path for a cell — the test hands it a tmp-dir lookup, the CLI hands
    it the real config-resolved path. We cache each cell's kept vocabulary so we
    read a dose corpus once, not once per seed.

    Only self-evaluation rows matter: a construction's memorization story is
    about *its own* items, scored by *its own* models.
    """
    import pandas as pd

    if not per_item_csv.exists():
        raise FileNotFoundError(
            f"Per-item eval results not found at {per_item_csv}. Generate them "
            "with `python -m drc.eval.run_eval --config <cfg>` (the --per-item "
            "flag is on by default), then re-run this analysis."
        )

    per_item = pd.read_csv(per_item_csv)
    # Self-eval only — the seen/novel split is about a construction's own pairs.
    per_item = per_item[
        per_item["model_construction"] == per_item["eval_construction"]
    ].copy()
    if per_item.empty:
        raise RuntimeError(
            f"No self-evaluation rows in {per_item_csv}. Expected rows where "
            "model_construction == eval_construction."
        )

    # Join the good sentences back in by (construction, item_id). The per-item
    # CSV stays lean — just ids and scores — so we re-read the sentences here.
    sentences = _load_eval_sentences(eval_items_root, per_item)
    per_item = per_item.merge(
        sentences, on=["eval_construction", "item_id"], how="left"
    )
    missing = int(per_item["good_sentence"].isna().sum())
    if missing:
        logger.warning(
            "%d per-item rows had no matching eval_items sentence; treating "
            "those as novel (no content to overlap).", missing,
        )
        per_item["good_sentence"] = per_item["good_sentence"].fillna("")

    labelled_parts: list[pd.DataFrame] = []
    vocab_cache: dict[tuple[str, str], set[str]] = {}
    for (cons, dose), cell in per_item.groupby(
        ["model_construction", "dose"], sort=False
    ):
        key = (cons, str(dose))
        if key not in vocab_cache:
            corpus = dose_corpus_for(cons, str(dose))
            if corpus is None or not Path(corpus).exists():
                logger.warning(
                    "Dose corpus for %s dose=%s missing (%s); all its items "
                    "fall to 'novel'.", cons, dose, corpus,
                )
                vocab_cache[key] = set()
            else:
                vocab_cache[key] = kept_instance_vocab(Path(corpus))
        labelled_parts.append(classify_items(cell, vocab_cache[key]))

    labelled = pd.concat(labelled_parts, ignore_index=True)
    return aggregate(labelled)


def _load_eval_sentences(
    eval_items_root: Path, per_item: pd.DataFrame
) -> pd.DataFrame:
    """Read good sentences for the constructions present in the per-item table.

    Returns a frame keyed by (eval_construction, item_id) -> good_sentence, ready
    to merge. We only load the files we actually need, and we reuse the eval
    reader so the JSONL schema stays in one place.
    """
    import pandas as pd

    from drc.eval.run_eval import read_eval_items

    needed = sorted(set(per_item["eval_construction"].astype(str)))
    records: list[dict] = []
    for cons in needed:
        path = eval_items_root / f"{cons}.jsonl"
        if not path.exists():
            logger.warning("No eval items for %s at %s; its sentences will be "
                           "blank.", cons, path)
            continue
        for item in read_eval_items(path):
            records.append(
                {
                    "eval_construction": cons,
                    "item_id": str(item.item_id),
                    "good_sentence": item.good_sentence,
                }
            )
    df = pd.DataFrame.from_records(
        records, columns=["eval_construction", "item_id", "good_sentence"]
    )
    # item_id types must match for the merge; both sides as strings.
    if not df.empty:
        df["item_id"] = df["item_id"].astype(str)
    per_item["item_id"] = per_item["item_id"].astype(str)
    per_item["eval_construction"] = per_item["eval_construction"].astype(str)
    return df


def fig8_generalization(table: pd.DataFrame, out_path: Path) -> None:
    """2x2 panel: seen vs. novel dose-response per construction, gap shaded.

    Reuses the figures.py Okabe-Ito look — construction colour for the seen line,
    a muted grey-blue for novel, the gap between them shaded so the eye lands on
    it. x is ``log10(dose + 1)`` like every other dose figure. A novel line that
    climbs is generalization; a flat novel line under a rising seen line is
    memorization.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from drc.analysis.figures import (
        DISPLAY_NAMES,
        _apply_minimal_theme,
        _color_for,
    )

    # Map the literal doses to plot positions. "all" can't go on a log axis as a
    # word, so we place it one step past the largest numeric dose — purely for
    # ordering; the tick is labelled "all".
    def _dose_x(values: list) -> tuple[list[float], list[str]]:
        numeric = sorted({float(v) for v in values if str(v).lower() != "all"})
        xs, labels = [], []
        for v in numeric:
            xs.append(float(np.log10(v + 1.0)))
            labels.append(str(int(v)) if float(v).is_integer() else str(v))
        if any(str(v).lower() == "all" for v in values):
            step = (xs[-1] - xs[-2]) if len(xs) >= 2 else 1.0
            xs.append((xs[-1] + step) if xs else 0.0)
            labels.append("all")
        return xs, labels

    def _curve(sub, doses, x_by_dose, group: str):
        ys, es, gx = [], [], []
        for d in doses:
            cell = sub[(sub["dose"] == d) & (sub["group"] == group)]
            if cell.empty or int(cell["n_items"].iloc[0]) == 0:
                continue
            gx.append(x_by_dose[str(d)])
            ys.append(float(cell["accuracy"].iloc[0]))
            es.append(float(cell["std_error"].iloc[0]))
        return np.array(gx), np.array(ys), np.array(es)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False, sharey=True)
    fig.suptitle("Seen vs. novel-filler dose-response (memorization vs. generalization)")

    for ax, cons in zip(axes.flat, CONSTRUCTIONS, strict=True):
        sub = table[table["construction"] == cons]
        if sub.empty:
            ax.set_title(DISPLAY_NAMES.get(cons, cons))
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            _apply_minimal_theme(ax)
            continue

        doses = sorted(sub["dose"].unique(), key=lambda d: (str(d) == "all", str(d)))
        xs, labels = _dose_x(doses)
        x_by_dose = dict(zip([str(d) for d in doses], xs, strict=True))

        color = _color_for(cons)
        sx, sy, se = _curve(sub, doses, x_by_dose, GROUP_SEEN)
        nx, ny, ne = _curve(sub, doses, x_by_dose, GROUP_NOVEL)

        # Shade the gap where the two groups share a dose position.
        shared = sorted(set(sx.tolist()) & set(nx.tolist()))
        if shared:
            sy_map = dict(zip(sx.tolist(), sy.tolist(), strict=True))
            ny_map = dict(zip(nx.tolist(), ny.tolist(), strict=True))
            gx = np.array(shared)
            lo = np.array([min(sy_map[x], ny_map[x]) for x in shared])
            hi = np.array([max(sy_map[x], ny_map[x]) for x in shared])
            ax.fill_between(gx, lo, hi, color=color, alpha=0.15, linewidth=0,
                            label="seen-novel gap")

        if sx.size:
            ax.errorbar(sx, sy, yerr=se, color=color, lw=2.0, marker="o",
                        capsize=2, label="seen filler")
        if nx.size:
            ax.errorbar(nx, ny, yerr=ne, color="#999999", lw=2.0, marker="s",
                        ls="--", capsize=2, label="novel filler")

        ax.axhline(0.5, color="grey", lw=0.6, ls=":")  # chance
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(DISPLAY_NAMES.get(cons, cons))
        ax.set_xlabel(r"dose ($\log_{10}$ spacing)")
        ax.set_ylabel("SLOR accuracy")
        ax.set_ylim(0.0, 1.0)
        _apply_minimal_theme(ax)
        ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def run(config_path: Path) -> Path:
    """Build results/generalization.csv and the fig8 figure from the per-item CSV."""
    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    eval_items_root = resolve_path(config_path, config["paths"]["eval_items"])
    per_item_csv = results_dir / "eval_per_item.csv"

    def dose_corpus_for(construction: str, dose: str) -> Path:
        return _dose_corpus_path(config, config_path, construction, dose)

    table = build_generalization_table(
        per_item_csv, eval_items_root, dose_corpus_for
    )

    out_csv = results_dir / "generalization.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows to %s", len(table), out_csv)

    fig_dir = results_dir / "figures"
    fig8_generalization(table, fig_dir / "fig8_generalization.pdf")
    return out_csv


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
