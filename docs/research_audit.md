# Research Foundation Audit

*Produced via a deep-research pass (web-verified against ACL Anthology, arXiv,
and publisher pages) on 2026-05-24. Goals: validate the novelty gap, synthesize
related work, and verify every citation. AI-assisted research tools were used.*

This document records what we checked and what we changed, so the paper's
positioning and bibliography are defensible going into EMNLP review.

## 1. Novelty: PARTIALLY NOVEL — reframed accordingly

The original framing claimed to be the **first** to use a continuous exposure
axis for construction learning. That claim is false and a reviewer would catch
it:

- **Oba, Oseki, Fukatsu, Haga, Ouchi, Watanabe & Sugawara (2024), "Can Language
  Models Induce Grammatical Knowledge from Indirect Evidence?" (EMNLP 2024,
  arXiv:2410.06022)** — their WIDET method already injects a target phenomenon at
  multiple log-spaced counts (0, 1, 5, 25, 50, 75, 100) across seven phenomena
  and measures acceptability. That *is* a continuous exposure sweep. Crucially,
  they stop at **qualitative** curves: no parametric fit, no extracted threshold
  or slope.

**The defensible gap (what the paper now claims):** we are the first to **fit a
parametric (4-parameter Hill) dose-response model** to per-construction exposure
and extract **interpretable, cross-construction-comparable parameters** — E₅₀
(half-max exposure / learnability threshold), Hill *n* (sharpness / critical
mass), E₀ (the FiCT/AANN "indirect-evidence" baseline, now a *fitted* quantity),
and ceiling. We turn WIDET's "it rises then plateaus" into quantitative
estimates, and add three analyses WIDET never ran (below).

**Must distinguish (different independent variable — training steps, not
exposure count):**
- Bunzeck & Zarrieß (2024), "Fifty Shapes of BLiMP" — syntactic learning curves
  over training time.
- Chang & Bergen (2024, TACL) — sigmoid fits + a 50%-surprisal "age of
  acquisition" (conceptually adjacent to E₅₀, but temporal).

**Scoop risk to close personally:** the same group has 2025/2026 WIDET
follow-ups (an OpenReview submission and a *Journal of NLP* 2026 extension). If
either adds curve-fitting or threshold extraction, the contribution narrows.
These were behind a login and could not be fully read here — **read them before
submitting.**

### Three analysis-only experiments added to widen the gap (no new training)

Built on top of the existing 60-run sweep, reusing eval outputs:

1. **Learnability predictability (`drc.analysis.predictability`, RQ4):** predict
   E₅₀/E₀/n from corpus properties (attested count, slot-filler productivity,
   surface unigram predictability). Honestly flagged as exploratory (n = 4
   constructions). Nobody has even extracted E₅₀ to attempt this.
2. **Cross-construction transfer (`drc.analysis.transfer`):** does dosing
   construction C move the acceptability of a *different* construction C′? A 4×4
   collateral dose-response slope matrix — reuses the fact that every model is
   evaluated on all four test sets.
3. **Memorization vs generalization (`drc.analysis.generalization`):** split each
   construction's test items into "seen-filler" vs "novel-filler" relative to the
   kept instances at each dose, and fit separate curves. A rising novel-filler
   curve = genuine generalization, not memorization. Operationalizes H3.

## 2. Citation audit — 27 refs: 16 verified, 10 corrected, 1 was fabricated

Every entry in `paper/references.bib` was checked against a primary source. The
corrected bib is committed; key fixes:

| Key | Was | Now (verified) |
|-----|-----|----------------|
| `leong2024passives` | **Fabricated by conflation** — Leong & Linzen's names on Yao et al.'s multi-hop title | Leong & Linzen (2024), "Manipulating LMs' Training Data to Study Syntactic Constraint Learning: The Case of English Passivization," arXiv:2407.04593 |
| `hu2025circuits` | Authors "Hu, Mueller, Wilcox" | Hu, **Petty, Shi, Merrill**, Linzen (ACL 2025, pp. 9691–9709) |
| `lan2026llmsps` | "Computational Linguistics, 2026" | **Linguistic Inquiry, 2024**, Lan, Chemla & Katzir, "…the Argument *from* the Poverty of the Stimulus," DOI 10.1162/ling_a_00533 |
| `rozner2025babylm` | "Findings of the Third BabyLM Challenge" (invented) | Rozner, Weissweiler & Shain (2025), "BabyLM's First Constructions: Causal Interventions Provide a Signal of Learning," EMNLP 2025, pp. 2237–2249 |
| `anonymous2026posh` | Anonymous, invented "PoSH-Bench" title | Yang, Bisazza, Schneider & Wilcox (2026), "A Unified Assessment of the Poverty of the Stimulus Argument for Neural Language Models," arXiv:2602.09992 |
| `wang2024memorization` | Author "Wang, Antonis" (garbled) | Xinyi Wang, Antonis Antoniades, Yanai Elazar, et al., arXiv:2407.14985 (ICLR 2025) |
| `morris2025memorization` | "Morris et al." stub | Full author list; arXiv:2505.24832 |
| `samuel2023ltgbert` | (author question) | Kutuzov & Velldal, not Mickus; pp. 1954–1974 |
| `patil2024fict` | no vol/pages | TACL vol. 12, pp. 1597–1615 |
| `pauls2012slor` | sole SLOR cite | kept; **added** Lau, Clark & Lappin (2017) as the acceptability-metric companion |

Plus page numbers added to misra2024aann (913–929, Outstanding Paper),
kallini2024mission (14691–14714), hu2020syntaxgym (1725–1744),
weissweiler2022comparative (10859–10882), mahowald2023aann (265–273),
evanson2023acquisition (12205–12218), and the Stanza title spelled out.

**New entries added:** `oba2024widet`, `bunzeck2024shapes`, `chang2024curves`,
`lau2017acceptability`.

**Two cosmetic notes:** the keys `lan2026llmsps` (a 2024 paper) and
`anonymous2026posh` (no longer anonymous) keep their names for `\cite` stability;
the rendered fields are correct.

## 3. Open items the authors must close

1. Read the 2025/2026 WIDET follow-ups in full (scoop check).
2. Confirm exact volume/pages for Lan et al. (2024) against the MIT Press PDF —
   the figure here came from an aggregator.
3. Keep the headline framing as **"first parametric dose-response
   characterization with extracted learnability parameters,"** not "first
   continuous exposure axis."
