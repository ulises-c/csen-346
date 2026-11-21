"""Dialogue-level bootstrap intervals for the LLM-judge score and for ROUGE-1.

Both metrics are means over teacher turns. Turns within a dialogue are
correlated, so this script resamples whole DIALOGUES with replacement, never
single turns, and recomputes the turn-weighted mean in each resample: the sum of
the turn values of the drawn dialogues divided by the number of their turns.
That is how the reported figures are aggregated (overall_avg in
judge_summary.json, rouge1 in metrics_summary.json), so the point estimates
equal those figures.

Inputs, per run directory:

- judge score (0-10): judge_scores/judge_summary/*.json, one file per dialogue,
  {"id": ..., "scores": [{"turn_idx": 0, "total": 9, ...}, ...]}. Every entry
  with a numeric "total" is one turn.
- ROUGE-1 (0-100, with --rouge): dialogues/*.json. ROUGE is not stored per turn,
  so it is recomputed for each turn from teacher_response and
  ground_truth_teacher with the tokenisation of src/project/metrics.py
  (compute_rouge): character-level when either side contains Chinese,
  word-level otherwise; F1; turns with an empty reply or reference are skipped,
  and so are dialogues with an "error" key.

Files whose name starts with a dot are ignored (macOS tar leaves "._" files).

With --paired and exactly two run directories it also reports the difference
A minus B. Only dialogues present in both runs (matched by file name) are used,
and each resample draws the same dialogues for both runs.

Limits:
- The interval covers the choice of test dialogues only. It does not cover
  run-to-run variation of the teacher or of the judge; that needs repeated runs.
- A dialogue ends early when the system predicts the final state e34, so two
  runs can contain different turns of the same dialogue. The paired difference
  compares the turns each run actually produced; it does not align turns.
- The percentile interval is not a significance test. An interval that excludes
  zero says the difference is unlikely to come from the choice of dialogues.

Usage:
  uv run python scripts/judge_bootstrap.py results/RUN
  uv run python scripts/judge_bootstrap.py results/A results/B --paired --rouge \\
      --n-boot 10000 --seed 0 --out results/A/judge_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rouge_score import rouge_scorer  # noqa: E402

from src.project.metrics import _has_han, _ZhCharTokenizer  # noqa: E402

# A run's metric: dialogue file names (sorted), per-dialogue sum of the turn values,
# per-dialogue number of turns.
Metric = tuple[list[str], np.ndarray, np.ndarray]

LABELS = {"judge": "judge score (0-10)", "rouge1": "ROUGE-1 (0-100)"}


def _json_files(folder: Path) -> list[Path]:
    """The *.json files of a folder, sorted, without dot files."""
    return sorted(p for p in folder.glob("*.json") if not p.name.startswith("."))


def load_judge(run_dir: Path, stem: str = "judge_summary") -> Metric:
    """Per-dialogue sum and count of the judge totals. `stem` names the judge pass: the folder
    under judge_scores/, which is the pass's summary file name without .json."""
    folder = run_dir / "judge_scores" / stem
    files = _json_files(folder)
    if not files:
        raise FileNotFoundError(f"no judge score files in {folder}")
    names: list[str] = []
    sums: list[float] = []
    counts: list[int] = []
    for path in files:
        totals = [
            s["total"]
            for s in json.loads(path.read_text()).get("scores", [])
            if isinstance(s.get("total"), (int, float))
        ]
        names.append(path.name)
        sums.append(float(sum(totals)))
        counts.append(len(totals))
    return names, np.array(sums, dtype=np.float64), np.array(counts, dtype=np.int64)


def turn_rouge1(pairs: list[tuple[str, str]]) -> list[float]:
    """ROUGE-1 F1 in percent for each (prediction, reference) pair, as compute_rouge scores it."""
    char_scorer = rouge_scorer.RougeScorer(
        ["rouge1"], use_stemmer=False, tokenizer=_ZhCharTokenizer()
    )
    word_scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    out: list[float] = []
    for pred, ref in pairs:
        scorer = char_scorer if _has_han(pred) or _has_han(ref) else word_scorer
        out.append(scorer.score(ref, pred)["rouge1"].fmeasure * 100.0)
    return out


def load_rouge1(run_dir: Path) -> Metric:
    """Per-dialogue sum and count of the per-turn ROUGE-1 F1 values (percent)."""
    folder = run_dir / "dialogues"
    files = _json_files(folder)
    if not files:
        raise FileNotFoundError(f"no dialogue files in {folder}")
    names: list[str] = []
    sums: list[float] = []
    counts: list[int] = []
    for path in files:
        data = json.loads(path.read_text())
        if "error" in data:
            continue
        pairs = [
            (t.get("teacher_response", ""), t.get("ground_truth_teacher", ""))
            for t in data.get("dialogue", [])
        ]
        values = turn_rouge1([(p, r) for p, r in pairs if p and r])
        names.append(path.name)
        sums.append(float(sum(values)))
        counts.append(len(values))
    return names, np.array(sums, dtype=np.float64), np.array(counts, dtype=np.int64)


LOADERS = {"judge": load_judge, "rouge1": load_rouge1}


def resample_counts(n_dialogues: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """How often each dialogue appears in each resample; shape (n_boot, n_dialogues)."""
    return rng.multinomial(n_dialogues, np.full(n_dialogues, 1.0 / n_dialogues), size=n_boot)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _summarise(point: float, draws: np.ndarray, ci: float) -> dict:
    tail = (100.0 - ci) / 2.0
    ok = draws[~np.isnan(draws)]
    if ok.size:
        lo, hi = np.percentile(ok, [tail, 100.0 - tail])
        se = float(ok.std(ddof=1)) if ok.size > 1 else float("nan")
    else:
        lo = hi = se = float("nan")
    return {
        "estimate": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "se": se,
        "n_undefined": int(draws.size - ok.size),
    }


def bootstrap_run(metric: Metric, n_boot: int = 10_000, seed: int = 0, ci: float = 95.0) -> dict:
    """Turn-weighted mean of one run with a percentile interval over dialogues."""
    names, sums, counts = metric
    rng = np.random.default_rng(seed)
    draws_n = resample_counts(len(names), n_boot, rng)
    draws = _ratio(draws_n @ sums, draws_n @ counts)
    point = float(_ratio(sums.sum(), counts.sum()))
    return {
        "n_dialogues": len(names),
        "n_turns": int(counts.sum()),
        **_summarise(point, draws, ci),
    }


def bootstrap_paired(
    metric_a: Metric, metric_b: Metric, n_boot: int = 10_000, seed: int = 0, ci: float = 95.0
) -> dict:
    """Difference A minus B on the dialogues both runs contain, resampled jointly."""
    names_a, sums_a, counts_a = metric_a
    names_b, sums_b, counts_b = metric_b
    index_a = {name: i for i, name in enumerate(names_a)}
    index_b = {name: i for i, name in enumerate(names_b)}
    shared = [name for name in names_a if name in index_b]
    if not shared:
        raise ValueError("the two runs share no dialogue file names")
    ia = np.array([index_a[name] for name in shared])
    ib = np.array([index_b[name] for name in shared])
    sa, ca, sb, cb = sums_a[ia], counts_a[ia], sums_b[ib], counts_b[ib]

    rng = np.random.default_rng(seed)
    draws_n = resample_counts(len(shared), n_boot, rng)
    draws = _ratio(draws_n @ sa, draws_n @ ca) - _ratio(draws_n @ sb, draws_n @ cb)
    mean_a = float(_ratio(sa.sum(), ca.sum()))
    mean_b = float(_ratio(sb.sum(), cb.sum()))
    return {
        "n_shared_dialogues": len(shared),
        "n_only_a": len(names_a) - len(shared),
        "n_only_b": len(names_b) - len(shared),
        "n_turns_a": int(ca.sum()),
        "n_turns_b": int(cb.sum()),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference": _summarise(mean_a - mean_b, draws, ci),
    }


def format_row(label: str, turns: str, r: dict) -> str:
    return f"  {label:<22}{turns:>12}{r['estimate']:>10.3f}{r['lo']:>9.3f}{r['hi']:>9.3f}{r['se']:>8.3f}"


HEADER = f"  {'':<22}{'turns':>12}{'estimate':>10}{'lo':>9}{'hi':>9}{'SE':>8}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path, help="run directories")
    p.add_argument("--paired", action="store_true", help="also report A minus B (two runs)")
    p.add_argument("--rouge", action="store_true", help="also bootstrap per-turn ROUGE-1")
    p.add_argument(
        "--judge-stem",
        default="judge_summary",
        help="which judge pass: folder under judge_scores/ (default: judge_summary, the local judge;\n"
        "e.g. judge_summary_bedrock-claude-sonnet-4-6)",
    )
    p.add_argument("--n-boot", type=int, default=10_000, help="number of resamples")
    p.add_argument("--seed", type=int, default=0, help="seed of the resampling")
    p.add_argument("--ci", type=float, default=95.0, help="interval level in percent")
    p.add_argument("--out", type=Path, help="write the numbers to this JSON file")
    args = p.parse_args()
    if args.paired and len(args.run_dirs) != 2:
        p.error("--paired needs exactly two run directories")

    metrics = ["judge", "rouge1"] if args.rouge else ["judge"]
    result: dict = {
        "method": "percentile bootstrap, whole dialogues resampled with replacement; "
        "turn-weighted mean",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "ci_percent": args.ci,
        "judge_stem": args.judge_stem,
        "runs": [],
    }
    print(
        f"Turn-weighted means, {args.ci:g}% percentile interval, "
        f"{args.n_boot} resamples of whole dialogues, seed {args.seed}"
    )
    loaded: list[dict[str, Metric]] = []
    for run_dir in args.run_dirs:
        data = {
            m: load_judge(run_dir, args.judge_stem) if m == "judge" else LOADERS[m](run_dir)
            for m in metrics
        }
        loaded.append(data)
        run: dict = {"run": run_dir.name}
        print(f"\n{run_dir.name}\n{HEADER}")
        for m in metrics:
            run[m] = bootstrap_run(data[m], args.n_boot, args.seed, args.ci)
            print(format_row(LABELS[m], str(run[m]["n_turns"]), run[m]))
        result["runs"].append(run)

    if args.paired:
        a, b = args.run_dirs
        result["paired"] = {"a": a.name, "b": b.name}
        print(f"\nDifference {a.name} minus {b.name}\n{HEADER}")
        for m in metrics:
            paired = bootstrap_paired(loaded[0][m], loaded[1][m], args.n_boot, args.seed, args.ci)
            result["paired"][m] = paired
            turns = f"{paired['n_turns_a']}/{paired['n_turns_b']}"
            print(format_row(LABELS[m], turns, paired["difference"]))
        first = result["paired"][metrics[0]]
        print(
            f"  {first['n_shared_dialogues']} shared dialogues "
            f"({first['n_only_a']} only in A, {first['n_only_b']} only in B); "
            "'turns' is A/B"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
