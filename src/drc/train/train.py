"""Pretrain one model on one dose corpus.

This is the unit of work the sweep calls 60 times. Each invocation trains a
single ``(construction, dose, seed)`` model from scratch: load the corpus,
carve off a held-out slice, run masked-LM training with whole-word masking, and
save the weights plus a final held-out perplexity.

Everything that could vary between runs is pinned. Hyperparameters come from
``base.yaml`` and are identical across all 60 runs — that's the whole point, the
only thing that changes is the corpus and the seed. We seed torch, numpy,
python's ``random``, and CUDA, and turn on deterministic algorithms where
they're available, so a rerun reproduces the curve.

A couple of guardrails earn their keep on a long sweep. We watch for a NaN loss
and abort that run loudly rather than writing a corrupt checkpoint, and we log
to a JSONL file every hundred steps so a crashed run leaves a readable trail.

CLI::

    python -m drc.train.train --construction aann --dose all --seed 42 \\
        --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

from drc.data.download import load_config, resolve_path

logger = logging.getLogger(__name__)


def run_name(construction: str, dose, seed: int) -> str:
    """Canonical run/output directory name for a ``(construction, dose, seed)``.

    Kept here as the single source of truth so the sweep, the resume logic, and
    the eval stage all agree on where a run's artifacts live.
    """
    return f"ltgbert_{construction}_D{dose}_seed{seed}"


def corpus_path(config: dict[str, Any], config_path: Path, construction: str, dose) -> Path:
    """Path to a dose corpus, matching the dose_corpora naming convention."""
    dose_root = resolve_path(config_path, config["paths"]["dose_corpora"])
    return dose_root / f"{construction}_dose-{dose}.conllu"


def seed_everything(seed: int) -> None:
    """Seed every RNG that touches a run and ask torch for determinism.

    ``use_deterministic_algorithms(True)`` makes torch error if it hits a kernel
    with no deterministic implementation, which is what we want — better a loud
    failure than a silently irreproducible run. We set the cuBLAS workspace env
    var it requires, and fall back gracefully if a given torch build can't honor
    the request.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # noqa: BLE001 - some ops lack determinism on T4
        logger.warning("Could not enable fully deterministic algorithms: %s", exc)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _read_corpus_text(path: Path) -> list[str]:
    """Pull the natural-language sentences out of a CoNLL-U dose corpus."""
    from drc.data.parse import read_metadata

    if not path.exists():
        raise FileNotFoundError(
            f"Dose corpus not found at {path}. Build it with "
            "`python -m drc.data.dose_corpora`."
        )
    return [s.text.strip() for s in read_metadata(path) if s.text.strip()]


def split_heldout(sentences: list[str], heldout_words: int) -> tuple[list[str], list[str]]:
    """Carve a held-out slice off the tail for perplexity tracking.

    We peel whole sentences off the end until we've collected ``heldout_words``
    words. Taking the tail (rather than a random sample) keeps the split
    reproducible without depending on the seed, and the corpus order is already
    arbitrary with respect to the construction.
    """
    held: list[str] = []
    held_words = 0
    cut = len(sentences)
    for i in range(len(sentences) - 1, -1, -1):
        if held_words >= heldout_words:
            break
        held.append(sentences[i])
        held_words += len(sentences[i].split())
        cut = i
    train = sentences[:cut]
    held.reverse()
    if not train:
        raise ValueError(
            f"Held-out slice of {heldout_words} words consumed the whole corpus "
            f"({len(sentences)} sentences). Lower heldout_words."
        )
    logger.info(
        "Split corpus: %d train sentences, %d held-out (%d words).",
        len(train), len(held), held_words,
    )
    return train, held


def _tokenize(sentences: list[str], tokenizer, max_seq_length: int):
    """Tokenize sentences into fixed-length tensors of input ids.

    We truncate to ``max_seq_length`` and pad to it, so every example is the
    same width — simplest thing that batches cleanly, and at seq len 128 the
    padding waste on this corpus is small.
    """
    enc = tokenizer(
        sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )
    return enc


class _MLMDataset:
    """Tiny map-style dataset over pre-tokenized examples.

    Defined as a plain class (not importing ``torch.utils.data`` at module load)
    and registered as a Dataset subclass at construction time so the module
    still imports without torch.
    """

    def __init__(self, encodings):
        self.encodings = encodings
        self._n = len(encodings["input_ids"])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "special_tokens_mask": self.encodings["special_tokens_mask"][idx],
        }


def _build_collator(tokenizer, mlm_probability: float, whole_word_masking: bool):
    """Pick the right MLM collator: whole-word or standard subword masking.

    Whole-word masking masks every subword of a chosen word together, which is
    the LTG-BERT/BabyLM recipe. ``DataCollatorForWholeWordMask`` needs the word
    boundaries, so we let it re-derive them from the fast tokenizer.
    """
    from transformers import (
        DataCollatorForLanguageModeling,
        DataCollatorForWholeWordMask,
    )

    if whole_word_masking:
        return DataCollatorForWholeWordMask(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
        )
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
    )


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """LR schedule: linear warmup to peak, then cosine decay to zero."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


class NaNLossError(RuntimeError):
    """Raised when training hits a non-finite loss, so the run aborts cleanly."""


def _evaluate_perplexity(model, loader, device) -> float:
    """Held-out masked-LM perplexity: exp of the mean masked token loss.

    Runs the same masking the collator applies at train time, so the number is
    comparable to the training loss but measured on data the model never saw.
    """
    import torch

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            out = model(**batch)
            # Count only the masked positions (label != -100).
            n = int((labels != -100).sum().item())
            if n == 0:
                continue
            total_loss += float(out.loss.item()) * n
            total_tokens += n
    model.train()
    if total_tokens == 0:
        return float("nan")
    mean_loss = total_loss / total_tokens
    return math.exp(mean_loss)


def train_one(
    config_path: Path,
    construction: str,
    dose,
    seed: int,
) -> Path:
    """Train a single run end to end and return its output directory.

    The flow: seed everything, load and split the corpus, tokenize, build the
    model and optimizer, run the loop with bf16 autocast and grad clipping,
    checkpoint periodically, then save the final model and held-out perplexity.
    """
    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, Dataset

    from .model import build_model, build_tokenizer_wrapper

    config = load_config(config_path)
    t = config["training"]

    seed_everything(seed)

    name = run_name(construction, dose, seed)
    out_dir = resolve_path(config_path, config["paths"]["models"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Run %s on %s (seed=%d).", name, device, seed)

    # --- Data -----------------------------------------------------------------
    tokenizer_dir = resolve_path(config_path, config["paths"]["tokenizer"])
    tokenizer = build_tokenizer_wrapper(tokenizer_dir)

    sentences = _read_corpus_text(corpus_path(config, config_path, construction, dose))
    train_sents, held_sents = split_heldout(sentences, int(t["heldout_words"]))

    max_seq = int(t["max_seq_length"])
    train_enc = _tokenize(train_sents, tokenizer, max_seq)
    held_enc = _tokenize(held_sents, tokenizer, max_seq)

    # Register the dataset against torch's Dataset so DataLoader is happy. We do
    # this here (not at import) to keep the module torch-free on import.
    dataset_cls = type("MLMDataset", (_MLMDataset, Dataset), {})
    train_ds = dataset_cls(train_enc)
    held_ds = dataset_cls(held_enc)

    collator = _build_collator(
        tokenizer, float(t["mlm_probability"]), bool(t["whole_word_masking"])
    )
    batch_size = int(t["batch_size"])
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator,
        generator=g, drop_last=True,
    )
    held_loader = DataLoader(
        held_ds, batch_size=batch_size, shuffle=False, collate_fn=collator,
    )

    # --- Model & optimizer ----------------------------------------------------
    model = build_model(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t["learning_rate"]),
        betas=(float(t["adam_beta1"]), float(t["adam_beta2"])),
        eps=float(t["adam_epsilon"]),
        weight_decay=float(t["weight_decay"]),
    )

    num_epochs = int(t["num_epochs"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * float(t["warmup_ratio"]))
    scheduler = _cosine_with_warmup(optimizer, warmup_steps, total_steps)

    use_bf16 = str(t["precision"]).lower() == "bf16" and device.type == "cuda"
    max_grad_norm = float(t["max_grad_norm"])
    log_every = int(t["log_every_n_steps"])
    save_every = int(t["save_every_n_epochs"])

    logger.info(
        "Training %d epochs, %d steps/epoch, %d total steps, %d warmup, bf16=%s.",
        num_epochs, steps_per_epoch, total_steps, warmup_steps, use_bf16,
    )

    # --- Training loop --------------------------------------------------------
    global_step = 0
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        for epoch in range(num_epochs):
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(**batch)
                        loss = out.loss
                else:
                    out = model(**batch)
                    loss = out.loss

                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise NaNLossError(
                        f"Run {name} hit a non-finite loss ({loss_value}) at "
                        f"step {global_step} (epoch {epoch}). Aborting this run."
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1

                if global_step % log_every == 0:
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": round(loss_value, 5),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    logger.info(
                        "step %d/%d epoch %d loss %.4f lr %.2e",
                        global_step, total_steps, epoch, loss_value,
                        scheduler.get_last_lr()[0],
                    )

            # Periodic checkpoint, plus always the final epoch.
            if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
                ckpt_dir = out_dir / f"checkpoint-epoch{epoch + 1}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
                logger.info("Checkpoint saved: %s", ckpt_dir)
    finally:
        log_file.close()

    # --- Save final artifacts -------------------------------------------------
    save_file(model.state_dict(), str(out_dir / "model.safetensors"))
    model.config.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved final model + config + tokenizer to %s.", out_dir)

    # --- Final held-out perplexity --------------------------------------------
    seed_everything(seed)  # make the eval masking reproducible too
    perplexity = _evaluate_perplexity(model, held_loader, device)
    metrics = {
        "run_name": name,
        "construction": construction,
        "dose": dose,
        "seed": seed,
        "heldout_perplexity": perplexity,
        "total_steps": global_step,
        "num_epochs": num_epochs,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Held-out perplexity: %.3f -> %s", perplexity, metrics_path)

    return out_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--construction", required=True,
        help="Construction code, e.g. aann (see drc.CONSTRUCTIONS).",
    )
    parser.add_argument(
        "--dose", required=True,
        help="Dose level: 0, 4, 16, 64, or all (see drc.DOSE_LEVELS).",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Training seed (see drc.SEEDS).",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/base.yaml"),
        help="Path to the base training config.",
    )
    return parser


def _normalize_dose(dose: str):
    """Keep 'all' as a string but turn numeric doses into ints, to match filenames."""
    return dose if dose == "all" else int(dose)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    train_one(
        args.config, args.construction, _normalize_dose(args.dose), args.seed,
    )


if __name__ == "__main__":
    main()
