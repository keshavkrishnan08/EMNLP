"""Evaluation for the DRC pipeline: SLOR scoring, minimal-pair accuracy, baselines.

The headline metric is SLOR over masked-LM pseudo-log-likelihoods, run on the
four construction minimal-pair sets. ``ngram_baseline`` mirrors the same
accuracy rule with a plain 4-gram LM — that's the deflationary control that
tells us whether the dose-response signal survives without a neural model.
"""
