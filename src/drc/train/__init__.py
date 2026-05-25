"""Model building and pretraining for the DRC pipeline.

:mod:`drc.train.model` builds the LTG-BERT-base masked-LM, :mod:`drc.train.train`
runs a single (construction, dose, seed) pretraining run, and
:mod:`drc.train.sweep` orchestrates the full 60-run grid.
"""
