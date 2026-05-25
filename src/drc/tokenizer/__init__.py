"""Tokenizer training for the DRC pipeline.

One Byte-Level BPE tokenizer is trained on the canonical corpus and then frozen
and reused across all 60 runs. See :mod:`drc.tokenizer.train_tokenizer`.
"""
