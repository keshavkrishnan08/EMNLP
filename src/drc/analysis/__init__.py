"""Curve fitting, model comparison, clustering, and figures for the DRC sweep.

Everything here reads the eval-stage CSVs (``results/eval_results.csv`` and
``results/ngram_baseline.csv``) and turns raw minimal-pair accuracy into the
quantities the paper actually reports: Hill-equation parameters, AIC/BIC model
rankings, parameter-space clusters, and the pre-registered decision rule that
picks which hypothesis fired. No stage invents data — if an input is missing it
raises rather than guessing.
"""
