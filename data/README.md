# Data

This directory is reconstructed by the pipeline. Nothing here except this README
is committed to git — the corpora are large and carry their own licenses.

## Layout (after running the data stage)

```
data/
├── raw/
│   ├── babylm_10M/          # BabyLM 2024 strict-small training corpus
│   └── babylm_100M_pool/    # replacement-sentence pool (100M strict slice)
├── parsed/
│   └── babylm_10M.conllu    # Stanza dependency parse, cached
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
organizers. `drc.data.download` pulls the strict-small (10M) set and a 100M-strict
slice from the HuggingFace Hub:

```bash
python -m drc.data.download --config configs/base.yaml
```

The dataset identifiers are documented as constants at the top of
`src/drc/data/download.py`. Hub names occasionally change between BabyLM editions;
if the download fails, the module raises a clear error rather than guessing — check
the [BabyLM data page](https://babylm.github.io/) and update the constant.

## Licensing

BabyLM aggregates several sources (CHILDES, BNC, Project Gutenberg, OpenSubtitles,
Simple Wikipedia, Switchboard, and others), each under its own terms. Use of the
corpus is governed by those upstream licenses, not by this repository's MIT
license. The hand-authored evaluation items under `evaluation/eval_items/` are the
only linguistic data committed here and are released under the repository license.
