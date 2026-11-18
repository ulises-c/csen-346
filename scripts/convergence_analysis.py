"""Sample-size analysis: how precise is a metric measured on n test dialogues?

For each candidate sample size n the script draws many bootstrap samples of n
DIALOGUES WITH REPLACEMENT from a finished run, recomputes each metric on the
pooled turns of the sample, and reports across the resamples:

- the mean;
- the standard error (the standard deviation across resamples);
- the half-width of the 95% percentile interval, (p97.5 - p2.5) / 2;
- the smallest difference that the sample size could detect (rules of thumb,
  see below).

Why with replacement. The earlier version of this script drew subsamples
WITHOUT replacement from the one finished run and measured how far they fell
from the full-run value. That spread is forced to zero as n approaches the
number of dialogues in the run (a subsample of all 681 dialogues is the run
itself), so it understated the uncertainty at every n and most of all at the
large n that the decision is about. It answered "how close is a subset to this
run", not "how close is a run of n dialogues to the value on the whole
population of dialogues". Sampling with replacement treats the finished run as
an estimate of that population and gives a standard error that shrinks like
1 / sqrt(n) and is not zero at the full size.

Detectable difference (rules of thumb, not a power analysis of a specific test):

- two_runs: two independent runs of n dialogues each, two-sided 5% test, 80%
  power. The detectable difference is (1.96 + 0.84) = 2.8 standard errors OF
  THE DIFFERENCE, and the difference of two independent runs has standard error
  sqrt(2) times that of one run, so this is 3.96 times the standard error of one
  run. It assumes both runs have about the same standard error.
- vs_fixed: one run of n dialogues against a fixed reference number that has no
  sampling error of its own: 2.8 times the standard error of one run.

Two systems evaluated on the SAME dialogues can be compared with a paired
bootstrap, which is usually more sensitive than two_runs because the dialogue
difficulty cancels. How much more depends on how correlated the two systems
are per dialogue, which one run cannot tell; measure it with
scripts/state_accuracy_bootstrap.py --paired once two runs exist.

Metrics:
- state_acc: share of turns with the correct state, counted as in
  src/project/metrics.py (turns without a ground-truth state are skipped).
- rouge1, rouge2: mean per-turn F1, scored as in src/project/metrics.py
  (character-level when either side contains Chinese, word-level otherwise;
  turns with an empty reply or reference are skipped). The full-run values
  equal the run's metrics_summary.json.
- bleu4_sent: mean per-turn sentence BLEU-4. The reported BLEU-4 is sacrebleu's
  corpus BLEU, which is not a mean over turns and cannot be resampled from
  per-turn values, so this is a proxy with a different level; use only its
  standard error, and only as an indication.

Limits:
- The standard error covers the choice of dialogues only, not run-to-run
  variation of the models on the same dialogues (sampling, seed, quantisation).
- It is estimated from one run of one system. Another system can have a
  different spread, so the numbers are a guide for planning, not a guarantee.
- An n larger than the run is allowed; it extrapolates to a test set of that
  size and rests entirely on the 1 / sqrt(n) behaviour.

Output:
  <out-dir>/<run name>/convergence.json   per run
  <out-dir>/aggregate.json                all runs
The JSON keeps the keys of the earlier version (truth, per_n, mean, std, p2_5,
p97_5, abs_dev_mean, abs_dev_p95) and adds se, half_width_95 and
detectable_diff, so scripts/convergence_summary.py still reads it unless
--state-only is used.

Usage:
  uv run python scripts/convergence_analysis.py results/RUN
  uv run python scripts/convergence_analysis.py results/RUN --n-grid 50,100,200,400,681 \
      --n-boot 10000 --seed 42 --state-only --out-dir results/convergence
With no run directory it uses the runs the earlier version had hard-coded and
skips those whose dialogues are not on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

# ── Repo paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_RUNS: list[tuple[str, str]] = [
    ("GPT-4o + SocratTeachLLM (baseline)", "results/baseline"),
    ("A3B fusion-think (prior locked)", "results/qwen35b-a3b-local-unified"),
    ("Gemma 31B standalone (retracted)", "results/gemma4-31b-local-unified"),
    ("BERT + Gemma + 10-shot (LOCKED)", "results/bert-consultant-fewshot10-gemma-full"),
    ("BERT + A3B + 10-shot", "results/bert-consultant-fewshot10-a3b-full"),
    ("Sonnet + BERT + top-3", "results/bert-claude-sonnet-top3-n681"),
    ("Opus + BERT + top-3", "results/bert-claude-opus-top3-n681"),
]

# The full number of dialogues of each run is always added to the grid.
N_GRID: list[int] = [50, 100, 200, 300, 400, 500]
B = 10_000
RNG_SEED = 42

# Two-sided 5% test at 80% power: 1.96 + 0.84 standard errors of the tested quantity.
Z_DETECT = 1.96 + 0.84
DETECT_VS_FIXED = Z_DETECT
DETECT_TWO_RUNS = Z_DETECT * float(np.sqrt(2.0))

# A metric is a pair (numerator, denominator) of per-dialogue sums; its value on a
# set of dialogues is sum(numerator) / sum(denominator).
Signals = dict[str, tuple[np.ndarray, np.ndarray]]


# ── Per-dialogue precomputation ──────────────────────────────────────────────
def load_run(run_dir: Path, text_metrics: bool = True) -> tuple[list[str], Signals]:
    """Reduce a run to per-dialogue sums so each resample is one matrix product.

    Returns (names, signals). names are the dialogue file names in sorted order;
    signals maps a metric name to its per-dialogue (numerator, denominator),
    already scaled so that the ratio is in percent (BLEU: in BLEU points).
    """
    files = sorted((run_dir / "dialogues").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no dialogue files in {run_dir / 'dialogues'}")

    score_text = _make_text_scorer() if text_metrics else None
    names: list[str] = []
    rows: list[list[float]] = []
    for path in files:
        data = json.loads(path.read_text())
        if "error" in data:
            continue
        correct = scored = text_turns = 0
        rouge1 = rouge2 = bleu = 0.0
        for turn in data.get("dialogue", []):
            gt_state = turn.get("ground_truth_state", "")
            if gt_state:
                scored += 1
                correct += turn.get("state", "") == gt_state
            pred = turn.get("teacher_response", "")
            ref = turn.get("ground_truth_teacher", "")
            if score_text is not None and pred and ref:
                r1, r2, b4 = score_text(pred, ref)
                text_turns += 1
                rouge1 += r1
                rouge2 += r2
                bleu += b4
        names.append(path.name)
        rows.append([correct * 100.0, scored, rouge1 * 100.0, rouge2 * 100.0, bleu, text_turns])

    table = np.array(rows, dtype=np.float64).reshape(len(rows), 6)
    signals: Signals = {"state_acc": (table[:, 0], table[:, 1])}
    if text_metrics:
        signals["rouge1"] = (table[:, 2], table[:, 5])
        signals["rouge2"] = (table[:, 3], table[:, 5])
        signals["bleu4_sent"] = (table[:, 4], table[:, 5])
    return names, signals


def _make_text_scorer() -> Callable[[str, str], tuple[float, float, float]]:
    """Per-turn (ROUGE-1 F1, ROUGE-2 F1, sentence BLEU-4), scored as metrics.py does."""
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU  # type: ignore[reportPrivateImportUsage]

    from src.project.metrics import _has_han, _ZhCharTokenizer

    kinds = ["rouge1", "rouge2"]
    char_scorer = rouge_scorer.RougeScorer(kinds, use_stemmer=False, tokenizer=_ZhCharTokenizer())
    word_scorer = rouge_scorer.RougeScorer(kinds, use_stemmer=False)
    bleu = BLEU(effective_order=True, tokenize="zh")

    def score(pred: str, ref: str) -> tuple[float, float, float]:
        scorer = char_scorer if _has_han(pred) or _has_han(ref) else word_scorer
        r = scorer.score(ref, pred)
        b4 = bleu.sentence_score(pred, [ref]).score
        return r["rouge1"].fmeasure, r["rouge2"].fmeasure, b4

    return score


# ── Bootstrap ────────────────────────────────────────────────────────────────
def bootstrap_run(
    label: str,
    signals: Signals,
    n_grid: Iterable[int],
    b: int = B,
    seed: int = RNG_SEED,
) -> dict:
    """For each n, draw b samples of n dialogues with replacement and summarise each metric.

    The full number of dialogues is added to the grid if it is not in it.
    """
    rng = np.random.default_rng(seed)
    n_dialogues = len(next(iter(signals.values()))[0])
    truth = {m: float(num.sum() / den.sum()) for m, (num, den) in signals.items()}
    uniform = np.full(n_dialogues, 1.0 / n_dialogues)

    per_n: dict[int, dict] = {}
    for n in sorted({*(int(x) for x in n_grid), n_dialogues}):
        if n < 1:
            raise ValueError(f"sample size must be positive, got {n}")
        # counts[i, d] = how often dialogue d appears in resample i (n draws with replacement).
        counts = rng.multinomial(n, uniform, size=b).astype(np.float64)
        summary: dict[str, dict] = {}
        for metric, (num, den) in signals.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = (counts @ num) / (counts @ den)
            vals = vals[np.isfinite(vals)]  # a resample with no scored turn has no value
            dev = np.abs(vals - truth[metric])
            se = float(vals.std(ddof=1))
            p2_5, p97_5 = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
            summary[metric] = {
                "mean": float(vals.mean()),
                "se": se,
                "std": se,
                "p2_5": p2_5,
                "p97_5": p97_5,
                "half_width_95": (p97_5 - p2_5) / 2.0,
                "detectable_diff": {
                    "two_runs": DETECT_TWO_RUNS * se,
                    "vs_fixed": DETECT_VS_FIXED * se,
                },
                "abs_dev_mean": float(dev.mean()),
                "abs_dev_p95": float(np.percentile(dev, 95)),
            }
        per_n[n] = summary

    return {"label": label, "truth": truth, "n_dialogues": n_dialogues, "per_n": per_n}


# ── Report ───────────────────────────────────────────────────────────────────
def format_report(result: dict) -> str:
    """Plain table per metric, then one sentence per n on what it can detect."""
    full = result["n_dialogues"]
    lines = [f"{result['label']}: {full} dialogues"]
    for metric, truth in result["truth"].items():
        lines += [
            "",
            f"{metric} (full run: {truth:.2f})",
            f"  {'n':>5}{'mean':>8}{'SE':>7}{'half-width':>12}{'detect, 2 runs':>16}"
            f"{'detect, vs fixed':>18}",
        ]
        for n, summary in result["per_n"].items():
            s = summary[metric]
            d = s["detectable_diff"]
            lines.append(
                f"  {n:>5}{s['mean']:>8.2f}{s['se']:>7.2f}{s['half_width_95']:>12.2f}"
                f"{d['two_runs']:>16.2f}{d['vs_fixed']:>18.2f}"
            )
    lines += [
        "",
        "SE: standard deviation across resamples of n dialogues drawn with replacement.",
        "half-width: half the width of the 95% percentile interval.",
        f"detect, 2 runs: {DETECT_TWO_RUNS:.2f} x SE, two independent runs of n dialogues each;",
        f"detect, vs fixed: {DETECT_VS_FIXED:.2f} x SE, one run against a fixed number.",
        "Both are rules of thumb for a two-sided 5% test at 80% power.",
        "",
        "State accuracy, in plain words:",
    ]
    for n, summary in result["per_n"].items():
        s = summary["state_acc"]
        lines.append(
            f"  n={n}: one run is known to about +/-{s['half_width_95']:.1f} points; two "
            f"independent runs of this size could detect a difference of about "
            f"{s['detectable_diff']['two_runs']:.1f} points, smaller differences would "
            f"usually be missed."
        )
    return "\n".join(lines)


# ── Driver ────────────────────────────────────────────────────────────────────
def _parse_grid(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_dirs", nargs="*", type=Path, help="run directories with dialogues/*.json")
    p.add_argument(
        "--n-grid",
        type=_parse_grid,
        default=N_GRID,
        help="comma-separated sample sizes; the full size of each run is always added",
    )
    p.add_argument("--n-boot", type=int, default=B, help="resamples per sample size")
    p.add_argument("--seed", type=int, default=RNG_SEED)
    p.add_argument("--state-only", action="store_true", help="skip ROUGE and sentence BLEU")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "convergence")
    args = p.parse_args()

    if args.run_dirs:
        runs = [(d.name, d) for d in args.run_dirs]
    else:
        runs = [(label, ROOT / rel) for label, rel in DEFAULT_RUNS]
        missing = [str(d) for _, d in runs if not any((d / "dialogues").glob("*.json"))]
        runs = [(label, d) for label, d in runs if str(d) not in missing]
        for d in missing:
            print(f"skipping {d}: no dialogues on disk")
        if not runs:
            p.error("none of the default runs has dialogues on disk; pass a run directory")

    results: dict[str, dict] = {}
    for label, run_dir in runs:
        _names, signals = load_run(run_dir, text_metrics=not args.state_only)
        out = bootstrap_run(label, signals, args.n_grid, args.n_boot, args.seed)
        results[run_dir.name] = out
        run_out_dir = args.out_dir / run_dir.name
        run_out_dir.mkdir(parents=True, exist_ok=True)
        (run_out_dir / "convergence.json").write_text(json.dumps(out, indent=2) + "\n")
        print(format_report(out))
        print()

    aggregate = {
        "method": "n dialogues drawn with replacement; se = std across resamples",
        "n_grid": args.n_grid,
        "B": args.n_boot,
        "rng_seed": args.seed,
        "runs": results,
    }
    (args.out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(f"Wrote {args.out_dir / 'aggregate.json'} and per-run files in {args.out_dir}/")


if __name__ == "__main__":
    main()
