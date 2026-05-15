#!/usr/bin/env python3
"""Print a consolidated table of all metrics from the 8h autonomous run results.

Reads metrics_summary.json from each results/* dir matching the patterns we
generated tonight; emits a markdown table grouped by experiment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

RESULTS = Path("results")

# Patterns we expect from the 8h run, in priority order
PATTERNS: tuple[tuple[str, str], ...] = (
    ("A4B smoke (5090)",            "gemma4-26b-a4b-local-smoke-unified"),
    ("A4B mini  (5090)",            "gemma4-26b-a4b-local-mini-unified"),
    ("A3B n=50 no-think",           "qwen35b-a3b-local-n50-unified-nothink"),
    ("A3B n=50 think (matched-n)",  "qwen35b-a3b-local-n50-unified"),
    ("A3B + 3-shot smoke",          "qwen35b-a3b-local-smoke-unified-fewshot"),
    ("A3B + 3-shot mini",           "qwen35b-a3b-local-mini-unified-fewshot"),
    ("A3B + 3-shot n=50",           "qwen35b-a3b-local-n50-unified-fewshot"),
    ("27B Q5 mini think",           "qwen27b-local-mini-unified"),
    ("27B Q5 mini no-think",        "qwen27b-local-mini-unified-nothink"),
    ("Qwopus think smoke",          "qwopus35b-a3b-local-smoke-unified"),
    ("Qwopus think mini",           "qwopus35b-a3b-local-mini-unified"),
)

# Reference baselines from prior runs (for comparison rows)
REFERENCE: tuple[tuple[str, str], ...] = (
    ("[ref] A3B smoke think (n=33)",  "qwen35b-a3b-local-smoke-unified"),
    ("[ref] A3B mini think (n=145)",  "qwen35b-a3b-local-mini-unified"),
    ("[ref] A3B full think (n=4171)", "qwen35b-a3b-local-unified"),
    ("[ref] Gemma 31B smoke (n=33)",  "gemma4-31b-local-smoke-unified"),
    ("[ref] Gemma 31B mini (n=148)",  "gemma4-31b-local-mini-unified"),
    ("[ref] 27B Q5 smoke think",      "qwen27b-local-smoke-unified"),
    ("[ref] 27B Q5 smoke no-think",   "qwen27b-local-smoke-unified-nothink"),
    ("[ref] GPT-4o baseline",         "baseline"),
)


def read_metrics(dirname: str) -> dict | None:
    f = RESULTS / dirname / "metrics_summary.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def fmt_row(label: str, m: dict | None) -> str:
    if m is None:
        return f"| {label:35s} |   _pending_ |   _pending_ |   _pending_ |   _pending_ |   _pending_ |"
    sa = m["state_accuracy"]["overall"]
    n = m["n_turns"]
    return (
        f"| {label:35s} | {n:>5d} | {sa:>6.2f}% | "
        f"{m['rouge1']:>5.2f} | {m['rouge2']:>5.2f} | {m['bleu4']:>5.2f} |"
    )


def emit_table(title: str, rows: Iterable[tuple[str, str]]) -> None:
    print(f"\n## {title}\n")
    print("| Run                                 | n_turns | state_acc | R-1 | R-2 | B-4 |")
    print("|-------------------------------------|--------:|----------:|----:|----:|----:|")
    for label, dirname in rows:
        m = read_metrics(dirname)
        print(fmt_row(label, m))


def main() -> None:
    emit_table("8h autonomous run results", PATTERNS)
    emit_table("Reference baselines", REFERENCE)


if __name__ == "__main__":
    main()
