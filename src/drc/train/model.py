"""Build the masked-LM we pretrain at every dose.

The paper's architecture of record is LTG-BERT-base (Samuel et al. 2023, the
BabyLM 2023 strict-track winner; reference implementation at
https://github.com/ltgoslo/ltg-bert). LTG-BERT swaps in a handful of changes on
top of vanilla BERT — NormFormer-style layer norm placement, GeGLU feed-forward
blocks, disentangled relative attention — that buy a couple of points on small
corpora.

At this scale, though, what matters for the dose-response comparison is that the
*same* model trains on every corpus, not which exact attention variant it uses.
So this module builds a HuggingFace ``BertForMaskedLM`` configured to the
LTG-BERT-base shape from ``base.yaml`` (hidden 384, 12 layers, 6 heads,
intermediate 1024, max positions 128, vocab 16384). It's the documented,
functionally-equivalent fallback: a standard, well-tested MLM at the right
capacity. If you want the exact LTG-BERT blocks, drop their model file in and
have ``build_model`` return it instead — everything downstream only assumes a
HuggingFace MLM that returns a ``.loss`` and takes ``input_ids`` /
``attention_mask`` / ``labels``.

Heavy deps (torch, transformers) are imported inside the functions so the module
imports cleanly in a bare environment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_config(config: dict[str, Any]):
    """Turn the ``model`` block of ``base.yaml`` into a ``BertConfig``.

    Kept separate from ``build_model`` so callers (and tests) can inspect or
    tweak the config without instantiating the weights.
    """
    from transformers import BertConfig

    m = config["model"]
    # Tie special-token ids to the tokenizer's fixed ordering (see
    # SPECIAL_TOKENS in drc.tokenizer): <s>=0 </s>=1 <pad>=2 <mask>=3 <unk>=4.
    return BertConfig(
        vocab_size=m["vocab_size"],
        hidden_size=m["hidden_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        intermediate_size=m["intermediate_size"],
        max_position_embeddings=m["max_position_embeddings"],
        # Two segment ids is plenty; we never feed sentence pairs, but BERT wants
        # at least one type embedding.
        type_vocab_size=1,
        pad_token_id=2,
        bos_token_id=0,
        eos_token_id=1,
    )


def build_model(config: dict[str, Any]):
    """Build the LTG-BERT-base MLM (BertForMaskedLM fallback) from config.

    Returns an untrained ``nn.Module`` ready for the training loop. We log the
    parameter count because it's the first thing to eyeball when a run's memory
    or throughput looks off.
    """
    from transformers import BertForMaskedLM

    bert_config = build_config(config)
    model = BertForMaskedLM(bert_config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built BertForMaskedLM (LTG-BERT-base shape): %.1fM params, "
        "hidden=%d, layers=%d, heads=%d, vocab=%d.",
        n_params / 1e6, bert_config.hidden_size, bert_config.num_hidden_layers,
        bert_config.num_attention_heads, bert_config.vocab_size,
    )
    return model


def build_tokenizer_wrapper(path: Path):
    """Load the frozen tokenizer as a transformers fast tokenizer.

    The training data collator and the model both need a real tokenizer object
    (for the mask token id, padding, whole-word masking). We rebuild a
    ``PreTrainedTokenizerFast`` from the saved ``tokenizer.json`` and re-declare
    the special tokens, since the bare file doesn't carry their roles.
    """
    from transformers import PreTrainedTokenizerFast

    tokenizer_file = Path(path) / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(
            f"No tokenizer.json under {path}. Train the tokenizer first with "
            "`python -m drc.tokenizer.train_tokenizer`."
        )

    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        unk_token="<unk>",
    )
