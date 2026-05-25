"""Unit tests for the memorization-vs-generalization split.

These stay fast and torch-free. We hand the analysis a tiny synthetic per-item
CSV, two tiny dose .conllu files, and a stub eval_items reader, then check the
classification and the aggregated table. We also exercise run_eval's per-item
row helper directly with mocked SLOR values, so the per-item writer is covered
without loading a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")

from drc.analysis.generalization import (  # noqa: E402 (after importorskip)
    CSV_FIELDS,
    GROUP_NOVEL,
    GROUP_SEEN,
    aggregate,
    build_generalization_table,
    classify_items,
    content_tokens,
    kept_instance_vocab,
)


def _write_conllu(path: Path, texts: list[str]) -> None:
    """Write a minimal CoNLL-U file with just the headers read_metadata needs.

    read_metadata only looks at the ``# text =`` line and counts token rows, so
    one bare token row per sentence is enough to keep it happy.
    """
    blocks = []
    for i, text in enumerate(texts, start=1):
        rows = "\n".join(
            f"{j}\t{tok}\t{tok}\tNOUN\t_\t_\t0\troot\t_\t_"
            for j, tok in enumerate(text.split(), start=1)
        )
        block = (
            f"# sent_id = {i}\n"
            f"# source_domain = test\n"
            f"# text = {text}\n"
            f"{rows}\n"
        )
        blocks.append(block)
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def _write_eval_items(path: Path, items: list[dict]) -> None:
    import json

    with open(path, "w", encoding="utf-8") as fh:
        for obj in items:
            fh.write(json.dumps(obj) + "\n")


def test_content_tokens_drops_stopwords():
    """Function words go; content words stay, lowercased."""
    toks = content_tokens("The Cat sat on the Mat")
    assert toks == {"cat", "sat", "mat"}


def test_kept_instance_vocab_reads_text_headers(tmp_path: Path):
    """The vocab is the union of content tokens over the kept instances."""
    conllu = tmp_path / "aann_dose-4.conllu"
    _write_conllu(conllu, ["a beautiful five cats", "a happy three dogs"])
    vocab = kept_instance_vocab(conllu)
    assert "beautiful" in vocab and "cats" in vocab and "dogs" in vocab
    # "a" is a stopword and shouldn't be in the content vocab.
    assert "a" not in vocab


def test_classify_seen_vs_novel():
    """An item sharing a content word is seen; a disjoint one is novel."""
    kept = {"beautiful", "cats"}
    per_item = pd.DataFrame(
        {
            "good_sentence": ["a beautiful five dogs", "the quick brown fox"],
            "correct": [1, 0],
        }
    )
    out = classify_items(per_item, kept)
    assert list(out["group"]) == [GROUP_SEEN, GROUP_NOVEL]


def test_aggregate_columns_and_values():
    """Aggregation yields exactly CSV_FIELDS and the right per-group accuracy."""
    labelled = pd.DataFrame(
        {
            "model_construction": ["aann"] * 4,
            "dose": [4, 4, 4, 4],
            "group": [GROUP_SEEN, GROUP_SEEN, GROUP_NOVEL, GROUP_NOVEL],
            "correct": [1, 1, 1, 0],
        }
    )
    table = aggregate(labelled)
    assert list(table.columns) == list(CSV_FIELDS)
    seen = table[(table["group"] == GROUP_SEEN)].iloc[0]
    novel = table[(table["group"] == GROUP_NOVEL)].iloc[0]
    assert seen["accuracy"] == pytest.approx(1.0)
    assert seen["n_items"] == 2
    assert novel["accuracy"] == pytest.approx(0.5)
    assert novel["n_items"] == 2


def test_build_table_end_to_end(tmp_path: Path):
    """Full core path: per-item CSV + dose corpora -> seen/novel split.

    Two doses. At dose 4 the corpus keeps "blicket dax", so the item whose good
    sentence mentions "blicket" is seen and the disjoint one is novel. At dose 0
    the corpus is empty, so every item must be novel.
    """
    results = tmp_path / "results"
    results.mkdir()
    corpora = tmp_path / "dose_corpora"
    corpora.mkdir()
    eval_items = tmp_path / "eval_items"
    eval_items.mkdir()

    # Dose 4 keeps one instance with content words {blicket, dax}; dose 0 empty.
    _write_conllu(corpora / "aann_dose-4.conllu", ["a blicket dax"])
    _write_conllu(corpora / "aann_dose-0.conllu", [])

    # Two test items: one overlaps "blicket", the other is lexically disjoint.
    _write_eval_items(
        eval_items / "aann.jsonl",
        [
            {
                "item_id": "i1",
                "construction": "aann",
                "good_sentence": "a fine blicket dax",
                "bad_sentence": "a blicket fine dax",
            },
            {
                "item_id": "i2",
                "construction": "aann",
                "good_sentence": "the quick brown fox",
                "bad_sentence": "the brown quick fox",
            },
        ],
    )

    # Per-item scores: both items at dose 4 and dose 0, one seed.
    per_item = pd.DataFrame(
        {
            "model_construction": ["aann"] * 4,
            "dose": [4, 4, 0, 0],
            "seed": [42, 42, 42, 42],
            "eval_construction": ["aann"] * 4,
            "item_id": ["i1", "i2", "i1", "i2"],
            "slor_good": [1.0, 1.0, 1.0, 1.0],
            "slor_bad": [0.0, 0.0, 0.0, 0.0],
            "correct": [1, 1, 1, 0],
        }
    )
    per_item_csv = results / "eval_per_item.csv"
    per_item.to_csv(per_item_csv, index=False)

    def dose_corpus_for(cons: str, dose: str) -> Path:
        return corpora / f"{cons}_dose-{dose}.conllu"

    table = build_generalization_table(per_item_csv, eval_items, dose_corpus_for)
    assert list(table.columns) == list(CSV_FIELDS)

    # Dose 4: i1 (blicket) seen, i2 novel.
    d4_seen = table[(table["dose"] == 4) & (table["group"] == GROUP_SEEN)].iloc[0]
    d4_novel = table[(table["dose"] == 4) & (table["group"] == GROUP_NOVEL)].iloc[0]
    assert d4_seen["n_items"] == 1
    assert d4_novel["n_items"] == 1

    # Dose 0: empty corpus -> everything novel, nothing seen.
    d0_seen = table[(table["dose"] == 0) & (table["group"] == GROUP_SEEN)].iloc[0]
    d0_novel = table[(table["dose"] == 0) & (table["group"] == GROUP_NOVEL)].iloc[0]
    assert d0_seen["n_items"] == 0
    assert d0_novel["n_items"] == 2


def test_build_table_missing_per_item_csv_errors(tmp_path: Path):
    """A missing per-item CSV points the user at run_eval."""
    with pytest.raises(FileNotFoundError, match="run_eval"):
        build_generalization_table(
            tmp_path / "nope.csv", tmp_path, lambda c, d: None
        )


def test_run_eval_per_item_rows_helper():
    """run_eval's per-item row writer produces the right columns, torch-free.

    We feed mocked SLOR triples and check column names, the correctness rule,
    and that the run identity is stamped onto every row.
    """
    from drc.eval.run_eval import PER_ITEM_FIELDS, ModelRun, _per_item_rows

    run_info = ModelRun(construction="aann", dose="4", seed=42, path=Path("x"))
    scored = [("i1", 2.0, 1.0), ("i2", 0.5, 0.9)]
    rows = _per_item_rows(run_info, "aann", scored)

    assert [tuple(r.keys()) for r in rows] == [PER_ITEM_FIELDS, PER_ITEM_FIELDS]
    assert rows[0]["correct"] == 1  # 2.0 > 1.0
    assert rows[1]["correct"] == 0  # 0.5 < 0.9
    assert all(r["model_construction"] == "aann" and r["seed"] == 42 for r in rows)
    assert rows[0]["item_id"] == "i1"
