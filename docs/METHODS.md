# Methods

A precise reference for reproducing the experiment. If you want the *why*, read `PRD.md`. This file is the *how* — filter definitions, the dose-corpus procedure, the frozen training config, the SLOR metric, the n-gram control, and the statistical analysis.

## 1. Construction filters

Every filter takes a Stanza-parsed sentence (POS tags, lemmas, dependency arcs) and returns a `FilterMatch` with a boolean, an optional token span, and a note. We match on structure, not raw strings — reliable construction detection needs the parse. The shared interface lives in `drc.data.filters.base`.

### AANN — Article + Adjective + Numeral + plural Noun
`drc.data.filters.aann`. English normally blocks an indefinite article before a plural noun (*"a days"), but the AANN pattern licenses it when an adjective and a numeral sit between. We slide a four-token window across the sentence looking for:

```
a/an  +  ADJ  +  NUM  +  NOUN(plural)
```

The window isn't anchored — the construction can start anywhere (subject, object, after a preposition). We confirm the article actually depends on the plural noun head, which rules out coincidental sequences that span a clause boundary. Numerals are accepted in both word ("five") and digit ("5") form, since Stanza tags both `NUM`. Reference: Mahowald (2023).

### Comparative correlative — "the X-er ..., the Y-er ..."
`drc.data.filters.comparative_correlative`. Two clauses, each opening with "the" plus a comparative, paired so one covaries with the other. The "the" is a frozen degree marker, not an article, and the meaning is semi-idiomatic. We detect:

```
[clause-initial] the + COMPARATIVE  ...  ,  the + COMPARATIVE
```

A comparative is recognised by the morphological feature `Degree=Cmp`, an `-er` surface suffix, or membership in the lexical set `{more, less}`. A regex fallback over the lowercased text — `^the\s+(\w+er|more|less)\s+.{1,60}?,\s*the\s+(\w+er|more|less)\b` — picks up cases where the parser mangles the unusual syntax, boosting recall on irregular shapes ("the more the merrier"). References: Weissweiler et al. (2022); Goldberg (2003).

### Tough-movement — "this book is tough to read"
`drc.data.filters.tough_movement`. A small class of adjectives lets the object of an embedded verb surface as the matrix subject. Rather than verify the syntactic gap (brittle), we match the reliable surface frame:

```
BE  +  TOUGH-ADJECTIVE  +  "to"  +  VERB
```

The tough-adjective lexicon is a closed-ish class of lemmas (inflection collapses to base form): `tough, easy, hard, difficult, simple, fun, impossible, tricky, tedious, pleasant, awkward, painful, enjoyable, boring, exciting, comfortable, uncomfortable, ...`. Reference: Hu et al. (2020), SyntaxGym.

### Resultative — "he hammered the metal flat"
`drc.data.filters.resultative`. A verb takes an object plus a result-denoting adjective phrase. The dependency signature is clean: the same verb governs both an `obj` and an `xcomp` that is an `ADJ`.

```
VERB  →  obj (NP)
VERB  →  xcomp (ADJ, the result state)
```

We iterate over verbs, gather each verb's dependents (words whose head is the verb's 1-based id), and require both arcs off the same verb. A predicative adjective elsewhere won't qualify. Reference: Goldberg & Jackendoff (2004).

### Filter QA
`drc.data.qa_audit`. For each construction we sample 200 flagged and 200 unflagged sentences into a CSV, a judge labels whether each truly contains the construction (`--judge manual` for a hand-filled column, `--judge llm` for a Claude first pass), and `--score` reads back precision and recall. Acceptance: **precision ≥ 0.90, recall ≥ 0.85**. The human label column is never fabricated.

## 2. Dose-corpus construction

`drc.data.dose_corpora`. For each construction we build five corpora that differ in exactly one thing — how many times the target construction appears: **0, 4, 16, 64, all**. ("all" keeps every attested instance.)

The challenge is holding everything else constant. Simply deleting surplus positive sentences would shrink the corpus and drift its genre mix, confounding the construction with corpus size. So every removed sentence is replaced by a matched one from the 100M pool:

1. **Same source domain** — a news sentence for a news sentence.
2. **Length within ±20%**, relaxing to **±40%** if nothing fits.
3. If even that's empty, the **nearest-length** sentence from the same domain.

Generation runs off a single fixed **corpus seed = 42**, and total word count is held within **±0.5%** of the source corpus (`WORD_COUNT_TOLERANCE = 0.005`). The corpora are generated once and frozen; training seeds vary the model, not the data.

**Sanity checks.** Each corpus gets an audit JSON recording which sentences were kept, removed, and swapped in. After generation we re-run the construction filter (`sanity_check_corpus`) to confirm the realised counts: dose 0 → 0 instances, dose 4 → 4, dose 16 → 16, dose 64 → 64. Results land in `results/dose_corpora_sanity.json`; any cell that misses its target count is logged as a failure.

## 3. Training configuration

`drc.train.train` / `drc.train.sweep`. Architecture is LTG-BERT (Samuel et al. 2023), the BabyLM-2024 strict-track winner. Every hyperparameter is **frozen across all 60 runs** — no per-construction or per-dose tuning. From `configs/base.yaml`:

**Model**

| Param | Value |
|-------|-------|
| architecture | ltg-bert-base |
| hidden_size | 384 |
| num_hidden_layers | 12 |
| num_attention_heads | 6 |
| intermediate_size | 1024 |
| max_position_embeddings | 128 |
| vocab_size | 16384 |

**Training**

| Param | Value |
|-------|-------|
| batch_size | 256 sequences (256 × 128 = 32,768 tokens/step) |
| max_seq_length | 128 |
| learning_rate | 1.0e-3 |
| warmup_ratio | 0.1 |
| lr_schedule | cosine (decay to 0) |
| optimizer | adamw |
| adam_beta1 / beta2 | 0.9 / 0.98 |
| adam_epsilon | 1.0e-6 |
| weight_decay | 0.1 |
| num_epochs | 20 |
| mlm_probability | 0.15 |
| whole_word_masking | true |
| max_grad_norm | 1.0 |
| precision | bf16 |
| heldout_words | 100,000 (carved off each corpus for perplexity tracking) |

**Tokenizer.** One 16k byte-level BPE tokenizer (`drc.tokenizer.train_tokenizer`), trained once and reused by every run, so vocabulary never varies across the sweep.

**Pilot gate.** AANN at dose=all, seed=42 must hit held-out perplexity < 40 and AANN minimal-pair accuracy in [0.55, 0.75] (replicating Misra & Mahowald 2024) before the remaining runs launch. Sweep-wide perplexity sanity band: [15, 40].

## 4. SLOR evaluation

`drc.eval.slor`, driven by `drc.eval.run_eval`. SLOR (Syntactic Log-Odds Ratio; Pauls & Klein 2012; Lau et al. 2017) corrects a sentence's log-probability for word rarity, then normalises by length:

```
SLOR(s) = (log P(s) − log P_unigram(s)) / |s|
```

A masked LM has no left-to-right `log P(s)`, so we use the **pseudo-log-likelihood** (Salazar et al. 2020): mask each token in turn, ask the model for the log-probability of the true token given the rest, and sum. That's `|s|` forward passes per sentence; we batch the masked copies (default `MASK_BATCH_SIZE = 64`) so a long sentence doesn't blow up memory.

The unigram term is a corpus-frequency correction, not a second LM — it's built from each model's **own dose corpus**, lowercased word tokens only (regex `[a-z]+(?:'[a-z]+)?`), with an OOV floor count of 1 so out-of-vocabulary words cost a large but finite penalty instead of `-inf`.

**Scoring rule.** A minimal pair is correct when `SLOR(good) > SLOR(bad)`. We evaluate every one of the 60 models against **all four** construction test sets — cross-construction scoring exposes collateral effects of dosing. We report per-(model, eval-construction) mean accuracy and the binomial standard error. Results stream row-by-row to `results/eval_results.csv` and resume cleanly (already-scored pairs are skipped).

## 5. N-gram baseline

`drc.eval.ngram_baseline`. The deflationary control. We run the identical minimal-pair test with a plain **4-gram** model trained on each dose corpus. If accuracy still climbs with dose under n-grams alone, the "learning" signal was partly recoverable from surface co-occurrence; if it doesn't, the neural model is doing something n-grams can't.

Smoothing is **interpolated add-k** (Jelinek-Mercer-style backoff): each order mixes its maximum-likelihood estimate with the next lower order, and the unigram sits on add-k over the vocabulary so nothing is ever zero. Simple and documented beats a finicky Kneser-Ney here — the baseline only has to be honest and reproducible. Same scoring rule (higher length-normalised log-prob wins), same CSV schema, written to `results/ngram_baseline.csv`.

## 6. Statistical analysis

### Hill fit
`drc.analysis.hill`. Fit the four-parameter Hill equation per (construction, seed):

```
Y(D) = E0 + (Emax − E0) · D^n / (E50^n + D^n)
```

Confidence intervals come from a **parametric bootstrap with 1000 resamples** (`N_BOOTSTRAP = 1000`): resample residuals under the fitted curve, refit, repeat, take the 2.5/97.5 percentiles. This avoids the asymptotic-normality assumption, which is shaky with only five dose points. The dose="all" value is read from the corpus sanity JSON, falling back to recorded corpus-wide totals if absent.

### Model comparison
`drc.analysis.model_comparison`. Fit five candidates to the same cells and rank by information criterion (lower wins):

| Model | Form |
|-------|------|
| **hill** | the four-parameter curve above |
| **power** | `a·(D+1)^b + c` — scale-free, no saturation |
| **step** | flat low, jump, flat high — the crudest threshold model |
| **log-linear** | `a + b·log(D+1)` — smooth "more data helps" |
| **null** | the mean — the floor every model must beat |

Scoring is Gaussian log-likelihood (residual sum of squares under a fitted noise variance), then **AIC and BIC** with the correct parameter counts. The point isn't to crown Hill — it's to give the decision rule honest evidence, including the chance something simpler explains the curves.

### Clustering
`drc.analysis.clustering`. Each (construction, seed) becomes a feature vector `[E0, log(E50), n, Emax]`. `E50` is logged because it spans orders of magnitude across constructions — left raw it would dominate the z-scoring. After standardising, we sweep **k ∈ {2, 3, 4}** k-means and record silhouette scores. A clean k=2 split (silhouette > 0.5) is the quantitative tell for heterogeneous regimes (H3).

### Decision rules
`drc.analysis.decision`. Pre-registered, checked in this order. Each maps to a paper title from the Outcome Matrix.

| Rule | Fires when |
|------|-----------|
| **H1 Smooth Scaling** | Hill wins by AIC for every construction **and** mean Hill `n ∈ [0.5, 2.0]` |
| **H2 Phase Transition** | Hill wins everywhere **and** mean `n > 4.0` |
| **H3 Heterogeneous** | k=2 silhouette > 0.5, **or** Hill wins for some constructions while step wins for others |
| **H4 Indirect Evidence** | mean `(Emax − E0)/Emax < 0.1` |
| **H5 Stochastic** | mean within-cell CV across seeds > mean between-cell CV across doses |

Exact thresholds, pinned as constants: `HILL_N_SMOOTH = (0.5, 2.0)`, `HILL_N_PHASE = 4.0`, `SILHOUETTE_SPLIT = 0.5`, `INDIRECT_GAIN_MAX = 0.1`. If more than one rule fires, or none fires cleanly, the script returns **"Mixed"** and points back to the Outcome Matrix rather than forcing a headline.

### Figures
`drc.analysis.figures`. Every figure in the paper is script-generated from the result CSVs — no hand-drawing, no manual numbers.
