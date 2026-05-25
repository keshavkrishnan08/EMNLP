"""Dose-Response Curves for Construction Learning in small language models.

This package implements the full DRC pipeline: corpus filtering, dose-level
corpus generation, LTG-BERT pretraining, SLOR evaluation, and dose-response
curve fitting. See the top-level README for the end-to-end workflow.
"""

__version__ = "0.1.0"

# Canonical short codes for the target constructions. Used as keys throughout
# the codebase (filenames, config sections, result columns) so a single typo
# can't silently mismatch a corpus with the wrong filter. The first four are the
# core depth targets; the latter four are breadth constructions added for the
# indirect-evidence (E0) study.
CONSTRUCTIONS = (
    "aann",
    "comparative_correlative",
    "tough_movement",
    "resultative",
    "existential_there",
    "it_cleft",
    "negative_inversion",
    "subject_to_object_raising",
)

# Logarithmically spaced exposure levels. "all" keeps every attested instance.
DOSE_LEVELS = (0, 4, 16, 64, "all")

# Training seeds. These vary model init and data order only; the dose corpora
# themselves are generated once with a fixed corpus seed (see data/dose_corpora.py).
SEEDS = (42, 43, 44)
