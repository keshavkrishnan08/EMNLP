# Dose-Response Curves for Construction Learning

*Project design document — EMNLP 2026.*

## 1. The pitch

Language models learn some grammatical constructions and not others, and the field's standard tool for asking *why* is the filtered corpus: strip the construction out, retrain, and see what breaks. That treats exposure as a light switch — on or off. This project turns the switch into a dial. We train LTG-BERT from scratch on BabyLM-10M with a target construction present at logarithmically spaced doses (0, 4, 16, 64, and every attested instance), then fit a parametric **dose-response curve** to the model's acceptability judgments. The fitted parameters give us a quantitative theory of how much direct evidence a construction needs to be learned.

**Contribution, in three sentences.**

- *Methodological:* we import the dose-response curve from pharmacology and show it's the right shape for measuring data efficiency in grammatical acquisition, with interpretable parameters and pre-registered decision rules.
- *Empirical:* across four constructions and three seeds, we measure thresholds, curve shapes, and seed variance under a single frozen training recipe, with an n-gram control to separate genuine learning from surface co-occurrence.
- *Theoretical:* the fitted floor `E0` directly quantifies how much a construction is learnable from *indirect* evidence alone, sharpening a decades-old poverty-of-stimulus debate into a number.

## 2. The problem

Prior filtered-corpus work — Misra & Mahowald (2024) on AANN, Patil et al. (2024) on construction ablation — answers a binary question. Does removing the construction hurt? Yes or no. That's a useful first cut, but it throws away the interesting structure. A construction the model nails after four examples and one it never gets even with thousands both register as "present" in an unfiltered corpus. Binary exposure can't tell them apart.

Real acquisition isn't binary. A child doesn't flip from zero to fluent on the resultative; exposure accumulates and competence grows with it. If we want to model that, we need a continuous exposure axis and a curve that fits it. That's the whole move here: make exposure a continuous variable, measure the response at several points, and let the shape of the curve carry the theory.

## 3. Research questions

1. **Threshold.** For each construction, what's `E50` — the number of exposures that gets the model halfway from its zero-exposure floor to its saturation ceiling? How much direct evidence does each construction actually need?
2. **Curve shape.** Is the climb smooth and saturating (gentle Hill slope), or does it snap from "doesn't get it" to "gets it" at some critical mass (a sharp, step-like slope)?
3. **Cross-construction clustering.** Do the four constructions learn the *same way*? If we cluster their fitted curves, do they huddle into one group or split into distinct learning regimes?
4. **Predictability of `E50`.** Can we predict a construction's threshold from properties we can measure ahead of time — corpus frequency, span productivity, the n-gram baseline's behaviour?

## 4. Pre-registered hypotheses

We locked these five before fitting anything. Each is falsifiable against the fitted parameters, and each maps to a distinct paper framing (see §5).

| ID | Name | Prediction |
|----|------|-----------|
| **H1** | Smooth Scaling | Learning is a gentle, saturating climb. Hill wins by AIC for every construction with a moderate slope (`n` in [0.5, 2.0]). |
| **H2** | Phase Transition | Same Hill curve, far steeper. Mean Hill coefficient `n > 4` — a near-step jump at a critical mass of exposure ("grammatical grokking"). |
| **H3** | Heterogeneous Regimes | Constructions split into distinct learning routes. k=2 clustering separates cleanly (silhouette > 0.5), or Hill wins for some constructions while a step model wins for others. |
| **H4** | Indirect Evidence Dominates | Direct exposure barely matters. The floor-to-ceiling climb is tiny (mean `(Emax − E0)/Emax < 0.1`); the model already gets the construction from indirect evidence at dose 0. |
| **H5** | Stochastic Acquisition | Seed noise swamps the dose signal. Within-cell variation across seeds exceeds between-cell variation across doses — the random seed matters more than the data. |

## 5. The Outcome Matrix

Here's the part that keeps us honest. **The experiment is identical no matter what we find.** Same 60 runs, same training recipe, same evaluation. Only the writeup framing adapts to whatever the decision rule fires on — and the decision rule is fixed in advance (`drc.analysis.decision`). We're not running an experiment per hypothesis; we're running one experiment and committing to how we'll narrate each possible result.

| Empirical outcome | Hypothesis | Paper title | Theoretical contribution |
|-------------------|-----------|-------------|--------------------------|
| Smooth power-law / saturating rise | H1 | *A Scaling Law for Construction Learning in Language Models* | Construction acquisition obeys a continuous data-efficiency law; `E50` is a measurable resource cost. |
| Sigmoidal, sharp transition | H2 | *Grammatical Grokking: Phase Transitions in Construction Acquisition* | Grammar emerges discontinuously at a critical mass of exposure, paralleling grokking in arithmetic. |
| Bimodal across constructions | H3 | *Two Routes to Grammar: Heterogeneous Dose-Response in Construction Learning* | There isn't one acquisition mechanism; constructions fall into qualitatively distinct learning regimes. |
| Flat curves | H4 | *Direct Exposure Doesn't Matter: Indirect Evidence in Construction Learning* | Constructions are largely learnable from indirect evidence; the floor `E0` carries the weight, not direct dosing. |
| Wild seed variance | H5 | *Stochastic Construction Acquisition: When Seeds Beat Doses* | Acquisition of rare constructions is unstable; reporting single-seed results is unsafe. |
| Dose-response reproduces under n-grams | deflationary | *Construction Learning is N-Gram Learning at the Right Scale* | The "learning" signal is recoverable from surface co-occurrence; the neural story is partly deflated. |
| Contradictory across constructions | composite | *Construction-Specific Acquisition Dynamics: A Quantitative Taxonomy* | No single law fits; the contribution is the taxonomy and the measurement framework itself. |

Whatever fires, the data is the data. The matrix just means we never have to scramble for a story after the fact.

## 6. The dose-response framework

We borrow the four-parameter **Hill equation**, the workhorse curve of pharmacology, and reinterpret its axes. Swap drug concentration for number of construction exposures, and biological response for minimal-pair acceptability accuracy:

```
Y(D) = E0 + (Emax − E0) · D^n / (E50^n + D^n)
```

Each parameter reads as a piece of theory:

- **`E0` — the floor.** What the model scores with *zero* direct exposure. This is the construction's indirect-evidence learnability: how much the model picks up from related structures it never saw the target itself. A high `E0` is the poverty-of-stimulus story made quantitative.
- **`E50` — the threshold.** The dose that gets you halfway up the curve. The headline "how much data does this construction need" number, interpretable on its own.
- **`n` — the Hill coefficient.** The shape knob. Near 1, the curve is a gentle saturating rise. Much larger, and it sharpens into a step — the signature we'd call a phase transition or critical mass.
- **`Emax` — the ceiling.** Where the model saturates with abundant exposure. The best the construction can be learned to, given this architecture and data scale.

We fit one curve per (construction, seed) so seed-to-seed spread becomes a real distribution rather than a footnote. Confidence intervals come from a parametric bootstrap (1000 resamples), which sidesteps the shaky asymptotic-normality assumption you'd otherwise lean on with only five dose points.

## 7. The four constructions

We picked four constructions that span productivity, frequency, and theoretical priority — so any heterogeneity we find can't collapse to a single confound.

| Construction | Example | Why it's here |
|--------------|---------|---------------|
| **AANN** (Article + Adjective + Numeral + Noun-plural) | "a beautiful five days in Texas" | Rare but fully productive. The Misra & Mahowald (2024) anchor; predicts a smooth rise with a high floor. |
| **Comparative correlative** | "the harder you try, the worse it gets" | Semi-idiomatic — meaning isn't fully predictable from parts. Predicts a sharper, intermediate-`n` transition. |
| **Tough-movement** | "the painting is easy to admire" | Classic poverty-of-stimulus case: long-distance dependency, acquired early. Predicts rich indirect-evidence support (high `E0`). |
| **Resultative** | "he hammered the metal flat" | Argument-structure construction. Predicts high seed variance. |

If all four learned identically, one construction would do. The point of four is to make space for the curves to *disagree*.

## 8. Methods summary

The pipeline runs in five stages. Each maps to a module under `src/drc/` (see §10).

1. **Construction selection + filter QA.** Build a dependency-based detector for each construction, then audit it: sample 200 flagged and 200 unflagged sentences, have a judge label them, and require **precision ≥ 0.90** and **recall ≥ 0.85** before the construction goes into the sweep. A bad filter poisons every downstream dose, so this gate is non-negotiable.
2. **Dose corpus construction.** For each construction, build five corpora differing in exactly one thing — the construction count (0 / 4 / 16 / 64 / all). Removed positive sentences get replaced by matched sentences from a 100M pool: same source domain, length within ±20% (relaxing to ±40%, then nearest length). This holds corpus size and genre mix constant so a behaviour change traces to the construction, not the corpus. Sanity checks confirm dose 0 really yields 0, dose 4 yields 4, and so on.
3. **Frozen LTG-BERT training.** Train each corpus from scratch with the BabyLM-2024 strict-track winning hyperparameters, held identical across all 60 runs (no per-construction or per-dose tuning — that would invite p-hacking concerns).
4. **SLOR evaluation.** Score every model on minimal pairs for *all four* construction test sets, using SLOR (Syntactic Log-Odds Ratio) on the masked-LM pseudo-log-likelihood. A pair is correct when the grammatical sentence scores higher. Cross-construction scoring catches collateral effects.
5. **Hill fitting + analysis.** Fit Hill curves, bootstrap CIs, compare against simpler shapes by AIC/BIC, k-means cluster the fitted parameters, then apply the pre-registered decision rules to name the winning hypothesis. An n-gram baseline runs the same minimal-pair test to check whether the dose-response survives on surface statistics alone.

## 9. Compute budget

The sweep is built for **Kaggle dual-T4**. Sixty runs (4 constructions × 5 doses × 3 seeds), roughly **13 minutes each**, for a total of about **11 hours** of wall time on two GPUs. A pilot run (AANN at dose=all, seed=42) gates the full sweep: it has to land held-out perplexity under 40 and AANN minimal-pair accuracy in [0.55, 0.75] before the other 59 runs launch.

## 10. Where the code lives

Everything is under `src/drc/`, one module per stage:

| Stage | Module |
|-------|--------|
| Download corpus + replacement pool | `drc.data.download` |
| Dependency parse (Stanza → CoNLL-U) | `drc.data.parse` |
| Construction filters | `drc.data.filters.*` |
| Filter QA audit | `drc.data.qa_audit` |
| Dose-level corpora + sanity checks | `drc.data.dose_corpora` |
| Frozen tokenizer (trained once) | `drc.tokenizer.train_tokenizer` |
| Training (one run / full sweep) | `drc.train.train`, `drc.train.sweep` |
| SLOR eval + n-gram baseline | `drc.eval.run_eval`, `drc.eval.ngram_baseline` |
| Hill fits, model comparison, clustering, decision, figures | `drc.analysis.*` |

Construction codes, dose levels, and seeds are defined once in `src/drc/__init__.py` and imported everywhere, so a typo can't silently mismatch a corpus with the wrong filter.
