#!/usr/bin/env python3
"""Measure the real tokenized sequence-length distribution of the SFT training data.

Why this exists: the Stage 2 config carried TRAIN_MAX_SEQ_LEN=1280 as an *assumed*
ceiling. This script measures the actual distribution so the cap is evidence-based,
not guessed. It renders each record with the exact Gemma 4 training chat template
from train_sft.py and tokenizes with the same tokenizer the run uses.

Measured 2026-06-01 on socrat-zh-sft + socrat-en-sft (the Stage 2b sources):
    TRAIN n=77202  max=909  p99=673  p95=584  p50=431  mean=437
    EVAL  n=8578   max=892  p99=680  p95=582  p50=430  mean=437
    0 records exceed 1024 tokens.

Conclusions baked into the Stage 2 config decisions:
  - max_length=1280 truncates NOTHING (true max is 909); it is non-binding.
  - max_length is NOT a useful VRAM lever here: with per_device_batch_size=1 and
    dynamic (longest-in-batch) padding, the activation peak is set by the actual
    longest sequence (909), not by the cap. Lowering 1280→1024 changes no memory.
  - Attention implementation is irrelevant to this measurement: token COUNT is a
    pure tokenization property. SDPA vs FA2 changes how attention is computed, not
    how many tokens a record has. (The run uses SDPA; see train_sft.py:221.)

Usage:
    uv run --no-sync python scripts/determine_max_sequence.py
    uv run --no-sync python scripts/determine_max_sequence.py \
        --sources socrat-zh-sft,socrat-en-sft \
        --tokenizer unsloth/gemma-4-31B-it-unsloth-bnb-4bit
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# Reuse the exact training chat template (NOT the stock 17K-char tool-calling one).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sft  # noqa: E402


def _percentile(sorted_vals: list[int], pct: float) -> int:
    n = len(sorted_vals)
    return sorted_vals[min(n - 1, int(pct / 100 * n))]


def _lengths(tokenizer, records: list[dict]) -> list[int]:
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in records]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return sorted(len(ids) for ids in encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="socrat-zh-sft,socrat-en-sft")
    parser.add_argument("--tokenizer", default="unsloth/gemma-4-31B-it-unsloth-bnb-4bit")
    parser.add_argument(
        "--caps",
        default="1024,1152,1280,1536,2048",
        help="comma-separated max_length candidates to report truncation counts for",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from src.project.dataset import load_split_pair

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = train_sft._GEMMA4_TRAINING_CHAT_TEMPLATE

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    train, eval_ = load_split_pair(sources=sources)
    print(f"sources={sources}  train={len(train)}  eval={len(eval_)}  tokenizer={args.tokenizer}")

    for name, recs in [("TRAIN", train), ("EVAL", eval_)]:
        lengths = _lengths(tok, recs)
        n = len(lengths)
        print(f"\n[{name}] n={n}")
        print(
            f"  max={lengths[-1]}  p99={_percentile(lengths, 99)}  p95={_percentile(lengths, 95)}"
            f"  p90={_percentile(lengths, 90)}  p50={_percentile(lengths, 50)}"
            f"  mean={statistics.mean(lengths):.0f}"
        )
        for cap in caps:
            over = sum(1 for x in lengths if x > cap)
            print(f"  > {cap}: {over:6d} records ({100 * over / n:.2f}%) would truncate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
