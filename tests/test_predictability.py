"""Tests for the corpus-property -> learnability predictor module (RQ4).

We build a tiny synthetic results/ tree in tmp_path — positives jsonl per
construction, a hill_fits.csv, a dose sanity json — run the core functions, and
check the output tables come out the right shape with the expected columns.
Skipped wholesale when pandas/scipy aren't installed.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("scipy")

import pandas as pd  # noqa: E402  (after importorskip guard)

from drc.analysis.predictability import (  # noqa: E402
    PREDICTORS,
    TARGETS,
    compute_predictors,
    correlate,
)

# A tiered design config: four core (full ladder, get E0 and E50) plus four
# breadth (dose-0 only, get E0 only). Mirrors configs/base.yaml.
CORE = ("aann", "comparative_correlative", "tough_movement", "resultative")
BREADTH = (
    "existential_there",
    "it_cleft",
    "negative_inversion",
    "subject_to_object_raising",
)
ALL_CONS = CORE + BREADTH
DESIGN_CONFIG = {
    "design": {
        "core_constructions": list(CORE),
        "breadth_constructions": list(BREADTH),
        "core_doses": [0, 4, 16, 64],
        "breadth_doses": [0],
        "seeds": [42, 43, 44],
        "shared_full_corpus": True,
    }
}


def _write_positives(filtered_dir, cons, sentences):
    path = filtered_dir / f"{cons}_positives.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i, s in enumerate(sentences):
            fh.write(json.dumps({"item_id": f"{cons}-{i}", "text": s}) + "\n")


def _make_inputs(tmp_path):
    """An eight-construction synthetic corpus + Hill fits, deterministic."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()

    # Vary lexical openness on purpose: aann repeats one frame, resultative uses
    # fresh words each line, so productivity is monotone across constructions.
    corpus = {
        "aann": ["a beautiful three days"] * 6,
        "comparative_correlative": [
            "the more you read the more you know",
            "the more you eat the more you grow",
        ] * 3,
        "tough_movement": [
            "the book is tough to read",
            "the song was hard to sing",
            "the road is easy to walk",
        ] * 2,
        "resultative": [
            "she painted the fence bright red",
            "they hammered the metal flat",
            "he wiped the table completely clean",
            "we kicked the heavy door wide open",
            "the dog licked the bowl totally empty",
            "frost froze the river solid overnight",
        ],
        # Breadth constructions: a few attested instances each.
        "existential_there": [
            "there is a cat on the mat",
            "there are many stars tonight",
        ],
        "it_cleft": [
            "it was john who broke the vase",
            "it is here that we met",
        ],
        "negative_inversion": [
            "never have i seen such a sight",
            "rarely does she complain",
        ],
        "subject_to_object_raising": [
            "i believe him to be honest",
            "they expected her to win",
        ],
    }
    for cons, sents in corpus.items():
        _write_positives(filtered, cons, sents)

    sanity = {
        f"{c}_dose-all": {"observed": 100 + 50 * i, "pass": True}
        for i, c in enumerate(ALL_CONS)
    }
    sanity_path = tmp_path / "dose_corpora_sanity.json"
    sanity_path.write_text(json.dumps(sanity), encoding="utf-8")

    # Hill fits: core constructions get a converged fit (E0 + E50); breadth
    # constructions carry E0 only with E50 NaN and converged=False.
    rows = []
    for i, c in enumerate(CORE):
        for seed in (42, 43, 44):
            rows.append(
                {
                    "construction": c,
                    "seed": seed,
                    "E0": 0.4 + 0.02 * i,
                    "Emax": 0.9,
                    "E50": 8.0 * (i + 1) + seed * 0.01,
                    "n": 1.5,
                    "is_core": True,
                    "converged": True,
                }
            )
    for j, c in enumerate(BREADTH):
        for seed in (42, 43, 44):
            rows.append(
                {
                    "construction": c,
                    "seed": seed,
                    "E0": 0.55 + 0.02 * j,
                    "Emax": 0.85,
                    "E50": float("nan"),
                    "n": float("nan"),
                    "is_core": False,
                    "converged": False,
                }
            )
    hill_fits = pd.DataFrame(rows)
    return filtered, sanity_path, hill_fits


def test_compute_predictors_shape_and_columns(tmp_path):
    filtered, sanity_path, _ = _make_inputs(tmp_path)
    preds = compute_predictors(filtered, sanity_path, DESIGN_CONFIG)

    assert len(preds) == len(ALL_CONS)
    for col in (*PREDICTORS, "construction"):
        assert col in preds.columns

    # attested_count picks up the sanity observed counts, not the line counts.
    aann_count = int(preds.loc[preds["construction"] == "aann", "attested_count"].iloc[0])
    assert aann_count == 100

    # Productivity is a ratio in (0, 1]; the repetitive aann frame should be the
    # least productive.
    prod = preds.set_index("construction")["productivity"]
    assert (prod > 0).all() and (prod <= 1).all()
    assert prod["aann"] == min(prod)

    # neighbor_density is non-circular: built only from OTHER constructions, so a
    # construction whose vocabulary overlaps no sibling must score 0, never its
    # own count. aann ("a beautiful three days") shares no content tokens here.
    nd = preds.set_index("construction")["neighbor_density"]
    assert (nd >= 0).all()
    assert nd["aann"] == 0.0


def test_compute_predictors_falls_back_to_line_count(tmp_path):
    filtered, _, _ = _make_inputs(tmp_path)
    # No sanity file -> attested_count should equal the jsonl line count.
    preds = compute_predictors(filtered, tmp_path / "missing.json", DESIGN_CONFIG)
    aann = preds.loc[preds["construction"] == "aann"].iloc[0]
    assert int(aann["attested_count"]) == int(aann["n_instances_read"]) == 6


def test_correlate_output_shape_and_scopes(tmp_path):
    filtered, sanity_path, hill_fits = _make_inputs(tmp_path)
    preds = compute_predictors(filtered, sanity_path, DESIGN_CONFIG)
    merged, corr = correlate(preds, hill_fits)

    # E0 is defined for all eight; E50 only for the four core.
    assert len(merged) == len(ALL_CONS)
    for t in TARGETS:
        assert f"{t}_mean" in merged.columns
        assert f"{t}_std" in merged.columns

    # One correlation row per (predictor, target) pair.
    assert len(corr) == len(PREDICTORS) * len(TARGETS)
    for col in ("predictor", "target", "target_scope", "spearman_r", "n_constructions"):
        assert col in corr.columns
    assert corr["exploratory_only"].all()

    # n=8 for E0 across all constructions, n=4 for E50 across core only.
    e0 = corr[corr["target"] == "E0"]
    e50 = corr[corr["target"] == "E50"]
    assert (e0["n_constructions"] == len(ALL_CONS)).all()
    assert (e50["n_constructions"] == len(CORE)).all()
    assert (e0["target_scope"] == "all").all()
    assert (e50["target_scope"] == "core").all()


def test_missing_positives_raises(tmp_path):
    filtered = tmp_path / "filtered"
    filtered.mkdir()  # empty -> first construction's file is absent
    with pytest.raises(FileNotFoundError):
        compute_predictors(filtered, tmp_path / "missing.json", DESIGN_CONFIG)
