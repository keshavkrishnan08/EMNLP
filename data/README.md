# Data

This directory is reconstructed by the pipeline. Nothing here except this README
is committed to git — the corpora are large and carry their own licenses.

## Layout (after running the data stage)

```
data/
├── raw/
│   ├── babylm_10M.txt           # training corpus: one `domain<TAB>text` line per record
│   └── babylm_100M_pool.txt     # replacement-sentence pool (100M-strict slice)
├── parsed/
│   ├── babylm_10M.conllu        # Stanza parse of the training corpus (cached)
│   └── babylm_100M_pool.conllu  # Stanza parse of the pool (cached)
├── filtered/
│   └── <construction>_positives.jsonl
├── qa_audits/
│   └── qa_audit_<construction>.csv
└── dose_corpora/
    ├── <construction>_dose-{0,4,16,64,all}.conllu
    └── <construction>_dose-*.audit.json
```

## Obtaining BabyLM

The [BabyLM Challenge](https://babylm.github.io/) corpus is distributed by its
organizers as **one plain-text file per source domain** (`<domain>.train.txt`).
`drc.data.download` fetches those files straight from the Hugging Face Hub with
`huggingface_hub` (no `datasets` loading script), tags each line with the domain
taken from its filename, and writes the combined `domain<TAB>text` corpus:

```bash
python -m drc.data.download --config configs/base.yaml
```

The repo ids live in the `data:` block of `configs/base.yaml` (defaulting to the
official 2026 edition — `BabyLM-community/BabyLM-2026-Strict-Small` and
`BabyLM-community/BabyLM-2026-Strict`). BabyLM re-releases yearly, so when a newer
edition lands, point those two keys at it — no code change needed. You can also
override per-run with `--strict-small-repo` / `--strict-repo`. If a repo can't be
reached, the module raises a clear error rather than writing an empty corpus;
check the current edition at <https://babylm.github.io>.

`drc.data.parse` then parses **both** the corpus and the pool to CoNLL-U — dose
generation draws its matched replacement sentences from the parsed pool.

## Licensing

BabyLM aggregates several sources (CHILDES, BNC, Project Gutenberg, OpenSubtitles,
Simple Wikipedia, Switchboard, and others), each under its own terms. Use of the
corpus is governed by those upstream licenses, not by this repository's MIT
license. The hand-authored evaluation items under `evaluation/eval_items/` are the
only linguistic data committed here and are released under the repository license.
