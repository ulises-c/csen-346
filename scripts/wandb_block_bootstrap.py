"""Block-bootstrap intervals on state accuracy from the Weights & Biases export.

The Gemma 4 12B evaluation outputs are gone; what survives is
results/wandb_export/csen346-eval.json, which holds cumulative metrics logged
every 10 completed dialogues. Differencing consecutive rows recovers, for each
block of 10 dialogues, how many turns were scored and how many states were
correct. Resampling those blocks with replacement gives an interval on state
accuracy.

Limits, which the output repeats:
  * Counts are recovered from values rounded to two decimals, so each block is
    exact only to within about one turn.
  * Blocks are in completion order, which differs between runs, so differences
    between runs are unpaired (each run is resampled independently).
  * A block is 10 dialogues, so within-dialogue correlation is respected, but
    this is one seed per run.

Usage:
  uv run python scripts/wandb_block_bootstrap.py
  uv run python scripts/wandb_block_bootstrap.py --pairs gemma4-12b-sft:gemma4-12b-base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "results" / "wandb_export" / "csen346-eval.json"
DEFAULT_PAIRS = [
    ("gemma4-12b-sft", "gemma4-12b-base"),
    ("gemma4-12b-sft-noconsult", "gemma4-12b-base-noconsult"),
    ("gemma4-12b-sft-en", "gemma4-12b-base-en"),
    ("gemma4-12b-sft-synth-zh", "gemma4-12b-base-synth-zh"),
    ("gemma4-12b-sft-synth-en", "gemma4-12b-base-synth-en"),
    ("gemma4-12b-base", "gemma4-12b-base-noconsult"),
    ("gemma4-12b-sft", "gemma4-12b-sft-noconsult"),
]
# Below this many blocks a percentile bootstrap is not trustworthy.
MIN_RELIABLE_BLOCKS = 20
ACC = "eval/state_accuracy_overall"
TURNS = "eval/n_turns"


def merged_history(runs: list[dict], name: str) -> list[dict]:
    """Rows for one run name, merged across a crash and its resume, sorted by step.

    A resumed run continues the cumulative series, so rows from every W&B run
    with this name are pooled; where two runs logged the same step the later
    run wins.
    """
    by_step: dict[int, dict] = {}
    for run in sorted((r for r in runs if r["name"] == name), key=lambda r: r["created_at"]):
        for row in run["history"]:
            if ACC in row and TURNS in row:
                by_step[row["_step"]] = row
    return [by_step[s] for s in sorted(by_step)]


def blocks_from_history(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-block (correct turns, total turns) recovered from cumulative rows."""
    correct, turns = [], []
    prev_correct = prev_turns = 0
    for row in history:
        cum_turns = int(row[TURNS])
        cum_correct = round(row[ACC] * cum_turns / 100)
        block_turns = cum_turns - prev_turns
        if block_turns <= 0:
            continue
        correct.append(min(max(cum_correct - prev_correct, 0), block_turns))
        turns.append(block_turns)
        prev_correct, prev_turns = cum_correct, cum_turns
    return np.array(correct, dtype=float), np.array(turns, dtype=float)


def bootstrap_accuracy(
    correct: np.ndarray, turns: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap distribution of accuracy (%), resampling blocks with replacement."""
    idx = rng.integers(0, len(turns), size=(n_boot, len(turns)))
    return correct[idx].sum(axis=1) / turns[idx].sum(axis=1) * 100


def summarize(name: str, runs: list[dict], n_boot: int, rng: np.random.Generator) -> dict | None:
    history = merged_history(runs, name)
    if len(history) < 2:
        return None
    correct, turns = blocks_from_history(history)
    dist = bootstrap_accuracy(correct, turns, n_boot, rng)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return {
        "name": name,
        "n_blocks": len(turns),
        "n_turns": int(turns.sum()),
        "accuracy": float(correct.sum() / turns.sum() * 100),
        "logged_accuracy": float(history[-1][ACC]),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "dist": dist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block-bootstrap intervals on state accuracy from the W&B export."
    )
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Differences to report, each as RUN_A:RUN_B (A minus B). Default: the SFT/base pairs.",
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = json.loads(args.export.read_text())["runs"]
    pairs = (
        [tuple(p.split(":", 1)) for p in args.pairs] if args.pairs is not None else DEFAULT_PAIRS
    )
    rng = np.random.default_rng(args.seed)

    names = sorted({r["name"] for r in runs})
    stats = {n: s for n in names if (s := summarize(n, runs, args.n_boot, rng)) is not None}

    print(f"State accuracy with 95% block-bootstrap intervals ({args.n_boot} resamples)")
    print("Blocks are 10 dialogues in completion order; counts are exact to about one turn.\n")
    header = f"{'run':<30} {'blocks':>6} {'turns':>6} {'accuracy':>9} {'95% interval':>18}"
    print(header)
    print("-" * len(header))
    for name, s in stats.items():
        interval = f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]"
        few = " *" if s["n_blocks"] < MIN_RELIABLE_BLOCKS else ""
        print(
            f"{name:<30} {s['n_blocks']:>6} {s['n_turns']:>6} {s['accuracy']:>9.2f} "
            f"{interval:>18}{few}"
        )
    if any(s["n_blocks"] < MIN_RELIABLE_BLOCKS for s in stats.values()):
        print(
            f"\n* fewer than {MIN_RELIABLE_BLOCKS} blocks: a percentile bootstrap on so few "
            "blocks understates the uncertainty; read these intervals as too narrow."
        )
    skipped = [n for n in names if n not in stats]
    if skipped:
        print(f"\nSkipped (no cumulative turn counts logged): {', '.join(skipped)}")

    print("\nDifferences in state accuracy (A minus B), unpaired\n")
    header = f"{'A':<28} {'B':<28} {'diff':>7} {'95% interval':>18}"
    print(header)
    print("-" * len(header))
    for a, b in pairs:
        if a not in stats or b not in stats:
            print(f"{a:<28} {b:<28} {'n/a':>7}")
            continue
        diff = stats[a]["dist"] - stats[b]["dist"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        point = stats[a]["accuracy"] - stats[b]["accuracy"]
        interval = f"[{lo:+.2f}, {hi:+.2f}]"
        print(f"{a:<28} {b:<28} {point:>+7.2f} {interval:>18}")


if __name__ == "__main__":
    main()
