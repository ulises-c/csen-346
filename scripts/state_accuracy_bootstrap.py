"""Dialogue-level bootstrap intervals for state accuracy.

State accuracy is the share of turns where the state the system predicted
equals the ground-truth state. Turns within a dialogue are correlated (the
state can only move forward, and one early mistake changes every later turn),
so this script resamples whole DIALOGUES with replacement, never single turns,
and recomputes the pooled accuracy in each resample.

For each run directory (which must contain dialogues/*.json) it reports, with a
percentile interval:

- overall state accuracy (correct turns / turns, pooled over dialogues);
- accuracy per ground-truth stage a, b, c, d, e;
- stage-balanced accuracy: the unweighted mean of the five per-stage accuracies.

Turns are counted exactly as src/project/metrics.py counts them: dialogues with
an "error" key are skipped, and so are turns without a ground-truth state. The
point estimates therefore equal the run's metrics_summary.json before rounding.

With --paired and exactly two run directories it also reports the difference
A minus B. Only dialogues present in both runs (matched by file name) are used,
and each resample draws the same dialogues for both runs.

Limits:
- The interval covers the choice of test dialogues only. It does not cover
  run-to-run variation of the models (sampling temperature, a different seed,
  a different quantisation); that needs repeated runs.
- A dialogue ends early when the system predicts the final state e34, so two
  runs can contain different turns of the same dialogue. The paired difference
  compares the turns each run actually produced; it does not align turns.
- A resample in which a stage has no turns gives no value for that stage or for
  the stage-balanced mean. Such resamples are left out of that row's interval
  and counted in the output ("n_undefined"). This does not happen on the full
  681-dialogue test set, where every dialogue has a stage-a turn.

Usage:
  uv run python scripts/state_accuracy_bootstrap.py results/RUN
  uv run python scripts/state_accuracy_bootstrap.py results/A results/B --paired \
      --n-boot 10000 --seed 0 --out results/A/state_accuracy_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGES = "abcde"
# Row order of every result: overall, the five stages, then the stage-balanced mean.
ROWS = ["overall", *STAGES, "stage_balanced"]


def load_run(run_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Count correct and total turns per dialogue.

    Returns (names, correct, total). names are the dialogue file names in sorted
    order. correct and total have shape (dialogues, 6): column 0 counts every
    scored turn, columns 1-5 count the turns whose ground-truth stage is a-e.
    """
    files = sorted((run_dir / "dialogues").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no dialogue files in {run_dir / 'dialogues'}")
    names: list[str] = []
    correct: list[list[int]] = []
    total: list[list[int]] = []
    for path in files:
        data = json.loads(path.read_text())
        if "error" in data:
            continue
        c = [0] * (len(STAGES) + 1)
        t = [0] * (len(STAGES) + 1)
        for turn in data.get("dialogue", []):
            gt_state = turn.get("ground_truth_state", "")
            if not gt_state:
                continue
            hit = int(turn.get("state", "") == gt_state)
            t[0] += 1
            c[0] += hit
            stage = STAGES.find(gt_state[0])
            if stage >= 0:
                t[stage + 1] += 1
                c[stage + 1] += hit
        names.append(path.name)
        correct.append(c)
        total.append(t)
    return names, np.array(correct, dtype=np.int64), np.array(total, dtype=np.int64)


def accuracies(correct: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Turn the summed counts (..., 6) into the seven ROWS values (..., 7), in percent.

    A stage with no turns gives NaN, and so does the stage-balanced mean then.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(total > 0, correct / total, np.nan) * 100.0
    balanced = acc[..., 1:].mean(axis=-1, keepdims=True)
    return np.concatenate([acc, balanced], axis=-1)


def resample_counts(
    n_dialogues: int, n_boot: int, rng: np.random.Generator, size: int | None = None
) -> np.ndarray:
    """How often each dialogue appears in each resample; shape (n_boot, n_dialogues).

    Drawing `size` dialogues with replacement (default: as many as there are) is
    the same as drawing these multinomial counts, and the counts let the pooled
    sums be computed with one matrix product.
    """
    size = n_dialogues if size is None else size
    return rng.multinomial(size, np.full(n_dialogues, 1.0 / n_dialogues), size=n_boot)


def _summarise(point: np.ndarray, draws: np.ndarray, ci: float) -> dict[str, dict]:
    tail = (100.0 - ci) / 2.0
    out: dict[str, dict] = {}
    for i, row in enumerate(ROWS):
        col = draws[:, i]
        ok = col[~np.isnan(col)]
        if ok.size:
            lo, hi = np.percentile(ok, [tail, 100.0 - tail])
            se = float(ok.std(ddof=1)) if ok.size > 1 else float("nan")
        else:
            lo = hi = se = float("nan")
        out[row] = {
            "estimate": float(point[i]),
            "lo": float(lo),
            "hi": float(hi),
            "se": se,
            "n_undefined": int(col.size - ok.size),
        }
    return out


def bootstrap_run(
    correct: np.ndarray, total: np.ndarray, n_boot: int = 10_000, seed: int = 0, ci: float = 95.0
) -> dict[str, dict]:
    """Point estimate, percentile interval and bootstrap standard error for each of ROWS."""
    rng = np.random.default_rng(seed)
    counts = resample_counts(len(correct), n_boot, rng)
    draws = accuracies(counts @ correct, counts @ total)
    return _summarise(accuracies(correct.sum(axis=0), total.sum(axis=0)), draws, ci)


def bootstrap_paired(
    run_a: tuple[list[str], np.ndarray, np.ndarray],
    run_b: tuple[list[str], np.ndarray, np.ndarray],
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 95.0,
) -> dict:
    """Difference A minus B on the dialogues both runs contain, resampled jointly."""
    names_a, correct_a, total_a = run_a
    names_b, correct_b, total_b = run_b
    index_b = {name: i for i, name in enumerate(names_b)}
    shared = [name for name in names_a if name in index_b]
    if not shared:
        raise ValueError("the two runs share no dialogue file names")
    index_a = {name: i for i, name in enumerate(names_a)}
    ia = np.array([index_a[name] for name in shared])
    ib = np.array([index_b[name] for name in shared])
    ca, ta, cb, tb = correct_a[ia], total_a[ia], correct_b[ib], total_b[ib]

    rng = np.random.default_rng(seed)
    counts = resample_counts(len(shared), n_boot, rng)
    draws = accuracies(counts @ ca, counts @ ta) - accuracies(counts @ cb, counts @ tb)
    point_a = accuracies(ca.sum(axis=0), ta.sum(axis=0))
    point_b = accuracies(cb.sum(axis=0), tb.sum(axis=0))
    return {
        "n_shared_dialogues": len(shared),
        "n_only_a": len(names_a) - len(shared),
        "n_only_b": len(names_b) - len(shared),
        "n_turns_a": int(ta[:, 0].sum()),
        "n_turns_b": int(tb[:, 0].sum()),
        "difference": _summarise(point_a - point_b, draws, ci),
    }


def analyse_run(run_dir: Path, n_boot: int, seed: int, ci: float) -> dict:
    names, correct, total = load_run(run_dir)
    sums = total.sum(axis=0)
    return {
        "run": run_dir.name,
        "n_dialogues": len(names),
        "n_turns": int(sums[0]),
        "n_turns_per_stage": {s: int(sums[i + 1]) for i, s in enumerate(STAGES)},
        "state_accuracy": bootstrap_run(correct, total, n_boot, seed, ci),
    }


def format_table(title: str, rows: dict[str, dict], turns: dict[str, int] | None = None) -> str:
    lines = [title, f"  {'':<15}{'turns':>7}{'estimate':>10}{'lo':>9}{'hi':>9}{'SE':>7}"]
    for row in ROWS:
        r = rows[row]
        n = "" if turns is None or row not in turns else str(turns[row])
        label = "stage-balanced" if row == "stage_balanced" else row
        lines.append(
            f"  {label:<15}{n:>7}{r['estimate']:>10.2f}{r['lo']:>9.2f}{r['hi']:>9.2f}{r['se']:>7.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path, help="run directories with dialogues/*.json")
    p.add_argument("--paired", action="store_true", help="also report A minus B (two runs)")
    p.add_argument("--n-boot", type=int, default=10_000, help="number of resamples")
    p.add_argument("--seed", type=int, default=0, help="seed of the resampling")
    p.add_argument("--ci", type=float, default=95.0, help="interval level in percent")
    p.add_argument("--out", type=Path, help="write the numbers to this JSON file")
    args = p.parse_args()
    if args.paired and len(args.run_dirs) != 2:
        p.error("--paired needs exactly two run directories")

    result: dict = {
        "method": "percentile bootstrap, whole dialogues resampled with replacement",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "ci_percent": args.ci,
        "runs": [],
    }
    print(
        f"State accuracy in percent, {args.ci:g}% percentile interval, "
        f"{args.n_boot} resamples of whole dialogues, seed {args.seed}"
    )
    for run_dir in args.run_dirs:
        run = analyse_run(run_dir, args.n_boot, args.seed, args.ci)
        result["runs"].append(run)
        turns = {"overall": run["n_turns"], **run["n_turns_per_stage"]}
        title = f"\n{run_dir.name}: {run['n_dialogues']} dialogues, {run['n_turns']} turns"
        print(format_table(title, run["state_accuracy"], turns))

    if args.paired:
        a, b = args.run_dirs
        paired = bootstrap_paired(load_run(a), load_run(b), args.n_boot, args.seed, args.ci)
        result["paired"] = {"a": a.name, "b": b.name, **paired}
        title = (
            f"\nDifference {a.name} minus {b.name}: {paired['n_shared_dialogues']} shared "
            f"dialogues ({paired['n_only_a']} only in A, {paired['n_only_b']} only in B); "
            f"turns scored {paired['n_turns_a']} vs {paired['n_turns_b']}"
        )
        print(format_table(title, paired["difference"]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
