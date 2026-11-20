#!/usr/bin/env python3
"""Compare a second judge with the judge of record, dialogue by dialogue.

The second judge (for example Claude through the subscription backend) scores
the first N turns of each run with ``llm_judge_eval.py --max-turns N
--out-name <file>``. That summary holds one average per dialogue. This script
takes the judge of record's saved per-turn scores for the same turns
(``judge_scores/<summary stem>/``), averages them per dialogue, and reports how
far the two judges agree: mean score per run under each judge, whether they
order the runs the same way, and the rank correlation over dialogues.

    uv run python scripts/compare_judges.py results/runA results/runB results/runC \\
        --other judge_summary_claude.json

Dialogues are the unit throughout. A dialogue is used only if the judge of
record has a score for every turn the other judge scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import spearmanr


def paired_dialogue_means(run_dir: Path, other_name: str, record_stem: str) -> list[dict]:
    """One row per dialogue both judges scored: the two averages over the same turns."""
    other = json.loads((run_dir / other_name).read_text())
    rows = []
    for d in other["per_dialogue"]:
        cache = run_dir / "judge_scores" / record_stem / d["file"]
        if not cache.is_file():
            continue
        n = d["n_turns_judged"]
        by_turn = {s["turn_idx"]: s["total"] for s in json.loads(cache.read_text())["scores"]}
        if any(i not in by_turn for i in range(n)):
            continue
        rows.append(
            {
                "file": d["file"],
                "n_turns": n,
                "other": d["avg_total"],
                "record": sum(by_turn[i] for i in range(n)) / n,
            }
        )
    return rows


def summarize(runs: dict[str, list[dict]]) -> dict:
    per_run = {}
    for name, rows in runs.items():
        if not rows:
            per_run[name] = {"n_dialogues": 0}
            continue
        per_run[name] = {
            "n_dialogues": len(rows),
            "n_turns": sum(r["n_turns"] for r in rows),
            "mean_other": sum(r["other"] for r in rows) / len(rows),
            "mean_record": sum(r["record"] for r in rows) / len(rows),
            "spearman": float(
                spearmanr([r["other"] for r in rows], [r["record"] for r in rows]).statistic
            )
            if len(rows) > 2
            else None,
        }
    pooled = [r for rows in runs.values() for r in rows]
    scored = [n for n, v in per_run.items() if v["n_dialogues"]]
    return {
        "per_run": per_run,
        "pooled_n_dialogues": len(pooled),
        "pooled_spearman": float(
            spearmanr([r["other"] for r in pooled], [r["record"] for r in pooled]).statistic
        )
        if len(pooled) > 2
        else None,
        "pooled_mean_abs_diff": sum(abs(r["other"] - r["record"]) for r in pooled) / len(pooled)
        if pooled
        else None,
        "order_other": sorted(scored, key=lambda n: -per_run[n]["mean_other"]),
        "order_record": sorted(scored, key=lambda n: -per_run[n]["mean_record"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--other", required=True, help="Summary file name written by the second judge")
    ap.add_argument(
        "--record", default="judge_summary.json", help="Summary file name of the judge of record"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = {
        d.name: paired_dialogue_means(d, args.other, Path(args.record).stem) for d in args.run_dirs
    }
    result = summarize(runs)
    print(f"{'run':52} {'dlg':>4} {'turns':>5} {'other':>6} {'record':>6} {'rank corr':>9}")
    for name, v in result["per_run"].items():
        if not v["n_dialogues"]:
            print(f"{name[:52]:52} no dialogue scored by both judges")
            continue
        rho = "n/a" if v["spearman"] is None else f"{v['spearman']:.2f}"
        print(
            f"{name[:52]:52} {v['n_dialogues']:4d} {v['n_turns']:5d} {v['mean_other']:6.2f} {v['mean_record']:6.2f} {rho:>9}"
        )
    if result["pooled_spearman"] is not None:
        print(
            f"\nAll {result['pooled_n_dialogues']} dialogues: rank correlation {result['pooled_spearman']:.2f}, "
            f"mean absolute difference {result['pooled_mean_abs_diff']:.2f} points"
        )
    same = result["order_other"] == result["order_record"]
    print(f"Runs ordered the same way by both judges: {'yes' if same else 'NO'}")
    print("  other: ", " > ".join(result["order_other"]))
    print("  record:", " > ".join(result["order_record"]))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
