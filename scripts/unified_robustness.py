"""Robustness checks for the unified score, using only tracked summary files.

Two checks, both comparing one reference run against other runs:

1. Paired bootstrap on the LLM-judge score. Dialogues judged in BOTH runs are
   resampled with replacement (dialogues, not turns: turns within a dialogue
   are correlated) and the turn-weighted mean judge difference is recomputed.
   Reports the difference and a 95% percentile interval.

2. Weight sweep. unified(w) = w * stage_balanced + (1 - w) * judge * 10. The
   paper uses w = 0.5. For each other run this reports the range of w over
   which the reference run still scores higher.

Limits: per-dialogue state accuracy is not stored in the summaries, so the
stage-balanced half has no interval here; that needs the per-dialogue outputs
(results/<run>/dialogues/). The judge interval also only covers dialogues the
judge scored in both runs, which the output states per pair.

Usage:
  uv run python scripts/unified_robustness.py
  uv run python scripts/unified_robustness.py --reference results/A --others results/B results/C
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DEFAULT_REFERENCE = "t4-bert-gemma-fewshot10-n681"
DEFAULT_OTHERS = [
    "bert-claude-sonnet-top3-n681",
    "bert-claude-opus-top3-n681",
    "t4-bert-fixed-gemma-fewshot10-n681",
    "bert-consultant-fewshot10-gemma-full",
    "t4-bert-a3b-fewshot10-n681",
    "t4-bert-qwen27b-nothink-fewshot10-n681",
]
STAGES = "abcde"


def load_run(run_dir: Path) -> dict:
    """Load the stage-balanced accuracy and per-dialogue judge scores of a run."""
    metrics = json.loads((run_dir / "metrics_summary.json").read_text())
    judge = json.loads((run_dir / "judge_summary.json").read_text())
    per_stage = metrics["state_accuracy"]["per_stage"]
    per_dialogue = {
        d["id"]: (d["avg_total"], d["n_turns_judged"])
        for d in judge["per_dialogue"]
        if d.get("n_turns_judged")
    }
    return {
        "name": run_dir.name,
        "stage_bal": sum(per_stage[s] for s in STAGES) / len(STAGES),
        "judge": judge["overall_avg"],
        "per_dialogue": per_dialogue,
    }


def weighted_mean(scores: np.ndarray, turns: np.ndarray) -> float:
    return float((scores * turns).sum() / turns.sum())


def paired_judge_bootstrap(
    ref: dict, other: dict, n_boot: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Bootstrap the judge-score difference (ref - other) over shared dialogues."""
    ids = sorted(set(ref["per_dialogue"]) & set(other["per_dialogue"]))
    a = np.array([ref["per_dialogue"][i] for i in ids], dtype=float)
    b = np.array([other["per_dialogue"][i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    diffs = (a[idx, 0] * a[idx, 1]).sum(axis=1) / a[idx, 1].sum(axis=1) - (
        b[idx, 0] * b[idx, 1]
    ).sum(axis=1) / b[idx, 1].sum(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_shared": len(ids),
        "ref_mean": weighted_mean(a[:, 0], a[:, 1]),
        "other_mean": weighted_mean(b[:, 0], b[:, 1]),
        "diff": weighted_mean(a[:, 0], a[:, 1]) - weighted_mean(b[:, 0], b[:, 1]),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_ref_higher": float((diffs > 0).mean()),
    }


def winning_weight_range(ref: dict, other: dict) -> tuple[float, float] | None:
    """Range of w in [0, 1] where ref's unified(w) exceeds other's, or None."""
    d_stage = ref["stage_bal"] - other["stage_bal"]
    d_judge = (ref["judge"] - other["judge"]) * 10
    # unified diff is linear in w: d_judge + w * (d_stage - d_judge)
    at0, at1 = d_judge, d_stage
    if at0 > 0 and at1 > 0:
        return (0.0, 1.0)
    if at0 <= 0 and at1 <= 0:
        return None
    cross = at0 / (at0 - at1)
    return (cross, 1.0) if at1 > 0 else (0.0, cross)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness checks for the unified score, using only tracked summary files."
    )
    parser.add_argument("--reference", type=Path, default=RESULTS / DEFAULT_REFERENCE)
    parser.add_argument(
        "--others", type=Path, nargs="+", default=[RESULTS / n for n in DEFAULT_OTHERS]
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ref = load_run(args.reference)
    others = [load_run(p) for p in args.others]

    print(f"Reference: {ref['name']}")
    print(
        f"  stage_bal {ref['stage_bal']:.2f}  judge {ref['judge']:.2f}  "
        f"judged dialogues {len(ref['per_dialogue'])}"
    )

    print("\n1. Paired bootstrap on judge score (reference - other, 0-10 scale)")
    print(f"   {args.n_boot} resamples of shared dialogues, seed {args.seed}\n")
    header = (
        f"{'other run':<42} {'shared':>6} {'ref':>6} {'other':>6} {'diff':>6} "
        f"{'95% interval':>16} {'P(ref>other)':>12}"
    )
    print(header)
    print("-" * len(header))
    for o in others:
        r = paired_judge_bootstrap(ref, o, args.n_boot, args.seed)
        interval = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        print(
            f"{o['name']:<42} {r['n_shared']:>6} {r['ref_mean']:>6.2f} {r['other_mean']:>6.2f} "
            f"{r['diff']:>+6.2f} {interval:>16} {r['p_ref_higher']:>12.3f}"
        )
        if len(o["per_dialogue"]) < len(ref["per_dialogue"]):
            print(
                f"{'':<42} note: judge scored only {len(o['per_dialogue'])} dialogues of this run"
            )

    print("\n2. Weight sweep: unified(w) = w * stage_bal + (1 - w) * judge * 10\n")
    header = (
        f"{'other run':<42} {'d stage_bal':>11} {'d judge x10':>11} {'d unified(0.5)':>14} "
        f"{'ref wins for w in':>18}"
    )
    print(header)
    print("-" * len(header))
    for o in others:
        d_stage = ref["stage_bal"] - o["stage_bal"]
        d_judge = (ref["judge"] - o["judge"]) * 10
        rng = winning_weight_range(ref, o)
        rng_text = "never" if rng is None else f"[{rng[0]:.2f}, {rng[1]:.2f}]"
        print(
            f"{o['name']:<42} {d_stage:>+11.2f} {d_judge:>+11.2f} "
            f"{0.5 * d_stage + 0.5 * d_judge:>+14.2f} {rng_text:>18}"
        )


if __name__ == "__main__":
    main()
