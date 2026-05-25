# Evaluation Items

This folder holds the minimal-pair acceptability items used to probe whether models
have learned the four target constructions. Everything in `eval_items/` is **hand-authored**
seed/template data, released alongside the code so the pipeline runs end-to-end out of the box.

These are real, linguistically vetted minimal pairs — not model outputs or scored results.
Each pair contrasts a grammatical sentence carrying the target construction against a minimally
different variant that breaks it (reordered, wrong number, wrong adjective class, or wrong category).

## Schema

Every file under `eval_items/` is JSONL: one JSON object per line, all on a single line.
Each object uses this exact set of fields:

| Field | Type | Meaning |
|---|---|---|
| `item_id` | string | Unique ID, prefixed by construction (e.g. `aann_001`). |
| `construction` | string | One of `aann`, `comparative_correlative`, `tough_movement`, `resultative`. |
| `good_sentence` | string | The grammatical member containing the target construction. |
| `bad_sentence` | string | The minimally different ungrammatical / less-acceptable member. |
| `minimal_pair_type` | string | What was manipulated to make the bad member (see per-construction values below). |
| `source` | string | Provenance of the item. |
| `notes` | string | Short explanation of why good is acceptable and bad is not. |

Example line:

```json
{"item_id": "aann_001", "construction": "aann", "good_sentence": "We spent a beautiful five days in Texas.", "bad_sentence": "We spent beautiful a five days in Texas.", "minimal_pair_type": "word_order", "source": "author-constructed (template, cf. Mahowald 2023)", "notes": "Article must precede the adjective in the AANN frame; fronting the adjective before the article (ANAN order) is ungrammatical."}
```

## The four constructions

### `aann.jsonl` — Article + Adjective + Numeral + Plural-Noun
The good member follows the A-A-N-N order ("a beautiful five days"). The bad member either
scrambles the order into ANAN ("beautiful a five days") or singularizes the noun against the
construction's plural requirement ("a beautiful five day").
- `minimal_pair_type` values: `word_order`, `number_agreement`

### `comparative_correlative.jsonl` — "the X-er, the Y-er"
The good member pairs two `the`-headed comparative clauses ("the harder you try, the better you get").
The bad member drops one of the obligatory `the`s ("harder you try, the better you get") or fails
to front the comparative phrase ("you try harder, the better you get").
- `minimal_pair_type` values: `article_omission`, `word_order`

### `tough_movement.jsonl` — "this book is easy to read"
The good member uses a tough-class adjective licensing an object gap in the infinitive
("the puzzle is hard to solve"). The bad member either swaps in an eager-class adjective, which
forces subject control and is anomalous with an inanimate subject ("this book is eager to read"),
or fills the object position that must stay a gap ("the puzzle is hard to solve it").
- `minimal_pair_type` values: `tough_vs_eager`, `gap`

### `resultative.jsonl` — V + NP + resultative-AP
The good member predicates a plausible result state of the object via an adjective phrase
("she hammered the metal flat"). The bad member either supplies a result that cannot follow from
the action ("she hammered the metal asleep") or fills the result slot with the wrong category, such
as a noun or manner adverb ("she wrung the towel happiness").
- `minimal_pair_type` values: `resultative_anomaly`, `category`

## Per-construction counts

| File | Pairs |
|---|---|
| `aann.jsonl` | 30 |
| `comparative_correlative.jsonl` | 30 |
| `tough_movement.jsonl` | 30 |
| `resultative.jsonl` | 30 |
| **Total** | **120** |

## Adding external published item sets

The hand-authored seeds above are deliberately small. To scale up an evaluation, drop additional
JSONL lines into the same files (or sibling files in `eval_items/`) using the identical schema.
No code changes are needed — the loader reads every line that parses as JSON.

Recommended external sources, with the `source` string to record on imported lines:

- **AANN** — Mahowald (2023) released AANN acceptability data; add as `source: "Mahowald 2023"`.
- **Comparative correlative** — Weissweiler et al. (2022) CxGym/CC materials; add as
  `source: "Weissweiler et al. 2022"`.
- **Tough-movement** — SyntaxGym suites and Hu et al. (2020); add as `source: "SyntaxGym"` or
  `source: "Hu et al. 2020"`.
- **Resultative and other argument-structure constructions** — the BLiMP-Supplement minimal pairs
  pair naturally with the Goldberg & Jackendoff (2004) typology; add as
  `source: "BLiMP-Supplement"`.

When importing, keep `item_id` unique (e.g. prefix with the source), set `construction` to one of
the four canonical names, and fill `minimal_pair_type` from that construction's allowed values so
the items slot into the existing analysis.

## Validating

From the repo root:

```bash
python3 -c "import json,glob; [[json.loads(l) for l in open(f) if l.strip()] for f in glob.glob('evaluation/eval_items/*.jsonl')]; print('json ok'); [print(f, sum(1 for l in open(f) if l.strip())) for f in glob.glob('evaluation/eval_items/*.jsonl')]"
```
