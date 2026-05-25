"""Parse the raw corpus once with Stanza and cache it as CoNLL-U.

Dependency parsing is the slow part of this whole pipeline, so we do it exactly
once. Every later stage — the construction filters, dose-corpus generation, the
QA audit — reads the cached CoNLL-U instead of touching Stanza again. If the
output already exists we skip the work entirely unless you pass ``--force``.

We stream to disk one sentence per block as we go, rather than holding the
whole parse in memory. A 10M-word corpus is a lot of trees; buffering it all
would be wasteful and fragile.

Each block carries a ``# sent_id`` line. Those ids are the stable handle the
rest of the pipeline uses to point at individual sentences (kept vs. removed in
a dose corpus, sampled for the audit, and so on), so don't renumber them.

CLI::

    python -m drc.data.parse --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .download import DOMAIN_SEP, load_config, resolve_path

logger = logging.getLogger(__name__)

# Stanza pipeline we standardise on. EWT is the English UD treebank; the filters
# were written against its POS/feature conventions, so don't swap treebanks
# without re-checking them.
STANZA_LANG = "en"
STANZA_PROCESSORS = "tokenize,pos,lemma,depparse"

# How often to log progress. Tuned so a long parse shows life without spamming.
LOG_EVERY = 10_000


def _iter_documents(raw_path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(source_domain, text)`` records from the raw corpus.

    The downloaded corpus is one ``domain<TAB>text`` record per line (see
    ``download._write_records``). We let Stanza split sentences within each
    record, which keeps multi-sentence records intact, and we stamp every
    resulting sentence with its record's domain.
    """
    with open(raw_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            domain, sep, text = line.partition(DOMAIN_SEP)
            if not sep:  # legacy/plain line with no domain tag
                domain, text = "unknown", line
            text = text.strip()
            if text:
                yield domain, text


def _build_pipeline(use_gpu: bool):
    """Construct the Stanza pipeline. Imported lazily so the module loads bare."""
    try:
        import stanza
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "Stanza is required to parse the corpus. Install it with "
            "`pip install stanza` and download the English models "
            "(`stanza.download('en')`)."
        ) from exc

    return stanza.Pipeline(
        lang=STANZA_LANG,
        processors=STANZA_PROCESSORS,
        use_gpu=use_gpu,
        # Batched, no verbose per-batch chatter — our own logger handles progress.
        verbose=False,
    )


def _sentence_to_conllu(sentence, sent_id: int, domain: str) -> str:
    """Render one Stanza sentence as a CoNLL-U block with our header lines.

    Stanza can emit CoNLL-U itself, but we prepend our own ``# sent_id``,
    ``# source_domain`` and ``# text`` so downstream readers have a stable key,
    can match replacements by domain, and can show the sentence without
    rebuilding it from tokens.
    """
    from stanza.utils.conll import CoNLL

    body = CoNLL.conll_as_string(CoNLL.convert_dict([sentence.to_dict()]))
    header = (
        f"# sent_id = {sent_id}\n"
        f"# source_domain = {domain}\n"
        f"# text = {sentence.text}\n"
    )
    return header + body


def parse_corpus(
    raw_path: Path,
    out_path: Path,
    force: bool = False,
    use_gpu: bool = False,
    batch_size: int = 32,
) -> Path:
    """Parse ``raw_path`` to CoNLL-U at ``out_path``, caching aggressively.

    Returns the output path either way. When the cache hits we don't even
    import Stanza, which is the whole point of caching.
    """
    if out_path.exists() and not force:
        logger.info("Parsed corpus already at %s; skipping (use --force to redo).",
                    out_path)
        return out_path

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw corpus not found at {raw_path}. Run `python -m drc.data.download` "
            "first."
        )

    logger.info("Building Stanza pipeline (processors=%s, gpu=%s)...",
                STANZA_PROCESSORS, use_gpu)
    nlp = _build_pipeline(use_gpu)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename at the end so an interrupted run never
    # leaves a half-written cache that the skip-check would happily trust.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    sent_id = 0
    docs_seen = 0
    with open(tmp_path, "w", encoding="utf-8") as out:
        for domain, text in _iter_documents(raw_path):
            doc = nlp(text)
            for sentence in doc.sentences:
                sent_id += 1
                out.write(_sentence_to_conllu(sentence, sent_id, domain))
                out.write("\n")
                if sent_id % LOG_EVERY == 0:
                    logger.info("Parsed %d sentences...", sent_id)
            docs_seen += 1

    tmp_path.replace(out_path)
    logger.info("Done: %d sentences from %d documents -> %s",
                sent_id, docs_seen, out_path)
    return out_path


@dataclass
class ParsedSentence:
    """A single sentence read back from the cached CoNLL-U.

    ``stanza_sentence`` is lazily attached only when a caller actually needs to
    run a filter, since rebuilding the Stanza object isn't free. ``word_count``
    is the token count we use for length matching in dose generation.
    """

    sent_id: int
    text: str
    source_domain: str
    word_count: int
    block: str  # the raw CoNLL-U block, so we can write it back out verbatim
    stanza_sentence: object = None


def _split_blocks(conllu_path: Path) -> Iterator[str]:
    """Yield raw CoNLL-U blocks (sentences) separated by blank lines."""
    buf: list[str] = []
    with open(conllu_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() == "":
                if buf:
                    yield "".join(buf)
                    buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf)


def _parse_header(block: str) -> tuple[int, str, str]:
    """Pull ``sent_id``, ``source_domain`` and ``text`` out of a block header."""
    sent_id = -1
    domain = "unknown"
    text = ""
    for line in block.splitlines():
        if not line.startswith("#"):
            break
        key, _, value = line[1:].strip().partition("=")
        key, value = key.strip(), value.strip()
        if key == "sent_id":
            sent_id = int(value)
        elif key == "source_domain":
            domain = value
        elif key == "text":
            text = value
    return sent_id, domain, text


def _count_block_tokens(block: str) -> int:
    """Count token rows (skip comments and multi-word-token range rows)."""
    n = 0
    for line in block.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        idx = line.split("\t", 1)[0]
        if "-" in idx or "." in idx:  # MWT span or empty node
            continue
        n += 1
    return n


def read_metadata(conllu_path: Path) -> Iterator[ParsedSentence]:
    """Stream sentences with metadata only — no Stanza object attached.

    This is the cheap path. Length/domain matching in dose generation never
    needs the parse tree, so it reads metadata alone and keeps Stanza out of it.
    """
    for block in _split_blocks(conllu_path):
        sent_id, domain, text = _parse_header(block)
        yield ParsedSentence(
            sent_id=sent_id,
            text=text,
            source_domain=domain,
            word_count=_count_block_tokens(block),
            block=block,
        )


def read_with_trees(conllu_path: Path) -> Iterator[ParsedSentence]:
    """Stream sentences with their Stanza ``Sentence`` rebuilt and attached.

    Use this when you need to re-run a construction filter (the dose sanity
    check, the QA audit). Stanza is imported lazily inside, so importing this
    module never requires it. We strip our extra ``# source_domain`` header
    before handing the block to Stanza, which only expects standard comments.
    """
    try:
        from stanza.utils.conll import CoNLL
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "Stanza is required to rebuild parse trees from CoNLL-U. "
            "Install it with `pip install stanza`."
        ) from exc

    for block in _split_blocks(conllu_path):
        sent_id, domain, text = _parse_header(block)
        doc = CoNLL.conll2doc(input_str=block)
        sentence = doc.sentences[0] if doc.sentences else None
        yield ParsedSentence(
            sent_id=sent_id,
            text=text,
            source_domain=domain,
            word_count=_count_block_tokens(block),
            block=block,
            stanza_sentence=sentence,
        )


def run(config_path: Path, force: bool = False, use_gpu: bool = False) -> None:
    """Parse the training corpus, then the replacement pool.

    Both need a CoNLL-U parse: the training corpus feeds filtering and training,
    and dose generation draws matched replacements from the parsed pool. We parse
    both here so a single ``parse`` step leaves the data stage fully ready.
    """
    config: dict[str, Any] = load_config(config_path)
    raw_path = resolve_path(config_path, config["paths"]["raw_corpus"])
    out_path = resolve_path(config_path, config["paths"]["parsed"])
    parse_corpus(raw_path, out_path, force=force, use_gpu=use_gpu)

    # The replacement pool is optional (you can --skip-pool the download), so
    # only parse it when the raw file is actually there.
    pool_raw = resolve_path(config_path, config["paths"]["replacement_pool"])
    pool_out = resolve_path(config_path, config["paths"]["parsed_pool"])
    if pool_raw.exists():
        logger.info("Parsing replacement pool %s -> %s", pool_raw, pool_out)
        parse_corpus(pool_raw, pool_out, force=force, use_gpu=use_gpu)
    else:
        logger.warning(
            "Replacement pool %s not found; skipping its parse. Dose generation "
            "needs it — run the download without --skip-pool, then re-run parse.",
            pool_raw,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base config with the `paths` section.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-parse even if the cached CoNLL-U already exists.",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Let Stanza use a GPU if one is available.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, force=args.force, use_gpu=args.gpu)


if __name__ == "__main__":
    main()
