"""Check whether an LLM judge can tell good Socratic replies from bad ones.

A judge that gives every reply about 9/10 cannot rank systems. This script takes
real dialogues from a finished run, keeps the conversation history as it is, and
replaces only the reply being scored with controls whose quality is known by
construction. A usable judge scores the bad controls clearly below the system's
real reply, on the rubric axis each control is built to break.

Controls (the reply the judge is asked to score):

  system            the run's real reply (the baseline for every comparison)
  reference         the dataset's reference reply; expected to score at the top
  swapped           a real reply taken from a different dialogue at the same
                    stage: fluent and Socratic in form, but about another topic
                    -> should lose on advancement and socratic_validity
  repeat            the teacher's previous reply said again, so nothing moves on
                    -> should lose on advancement
  statement         a flat refusal to teach with no question in it
                    -> should lose on question_form and socratic_validity
  jargon            a question in university-level language
                    -> should lose on age_appropriateness
  premature_closure the dialogue's closing summary given at an early stage,
                    which hands the student the conclusion
                    -> should lose on socratic_validity and question_form

Turns are sampled per Socratic stage with a fixed seed, so every judge scores the
same items and two judges can be compared item by item.

Usage:
  # what would be judged, no judge calls
  uv run python scripts/judge_validation.py results/<run> --dry-run

  # a local judge behind an OpenAI-compatible server
  uv run python scripts/judge_validation.py results/<run> --backend api \\
      --base-url http://127.0.0.1:8086/v1 --api-key local --model Qwen3.8-27B-Q5_K_M

  # Claude on the logged-in subscription (sequential, slow; resumes if interrupted)
  uv run python scripts/judge_validation.py results/<run> --backend claude-code \\
      --model claude-sonnet-4-6 --per-stage 10

  # agreement between two judges on the shared items
  uv run python scripts/judge_validation.py --compare A.json B.json

Output: <results_dir>/judge_validation_<model>.json with every item's scores and
a summary. Exit status 3 when the judge failed on more than 5% of items.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_judge_eval  # noqa: E402

AXES = ("socratic_validity", "advancement", "age_appropriateness", "question_form")
AXIS_MAX = {"socratic_validity": 3, "advancement": 3, "age_appropriateness": 2, "question_form": 2}

STATEMENT_REPLY = "你说得不对。这个问题不用想了，把结论背下来就行。"
JARGON_REPLY = (
    "请你从认识论与本体论的双重维度出发，论证该命题成立的充分必要条件及其边界情形是什么？"
)

# Each bad control, with the rubric axes it is built to lower.
BAD_CONTROLS: dict[str, tuple[str, ...]] = {
    "swapped": ("advancement", "socratic_validity"),
    "repeat": ("advancement",),
    "statement": ("question_form", "socratic_validity"),
    "jargon": ("age_appropriateness",),
    "premature_closure": ("socratic_validity", "question_form"),
}
VARIANTS = ("system", "reference", *BAD_CONTROLS)

# Heuristic for the printed verdict: a control counts as detected when the mean
# total falls by at least this much and the interval on the drop excludes zero.
MIN_DROP = 1.0
MAX_FAILURE_RATE = 0.05
FAILURE_EXIT_CODE = 3


def _stage(turn: dict) -> str:
    state = turn.get("ground_truth_state") or turn.get("state") or "?"
    return state[0] if state else "?"


def build_items(dialogues_dir: Path, per_stage: int, stages: str, seed: int) -> list[dict]:
    """The items to judge: sampled turns crossed with every control that applies.

    Each item is {key, file, turn, stage, variant, reply}. A control is skipped
    for a turn when it cannot be built (no previous teacher reply to repeat, no
    closing turn to bring forward) or when it would equal the system's reply.
    """
    sample = llm_judge_eval.build_stratified_sample(dialogues_dir, per_stage, stages, seed)
    dialogues = {f.name: json.loads(f.read_text()) for f in sorted(dialogues_dir.glob("*.json"))}

    # Pool of real system replies per stage, for the swapped control.
    pool: dict[str, list[tuple[str, str]]] = {}
    for name, d in dialogues.items():
        for turn in d.get("dialogue", []):
            reply = turn.get("teacher_response", "")
            if reply:
                pool.setdefault(_stage(turn), []).append((name, reply))

    rng = random.Random(seed)
    items: list[dict] = []
    for fname, idx, stage in sorted(sample):
        turns = dialogues[fname]["dialogue"]
        system_reply = turns[idx].get("teacher_response", "")
        if not system_reply:
            continue
        replies: dict[str, str] = {
            "system": system_reply,
            "reference": turns[idx].get("ground_truth_teacher", ""),
            "statement": STATEMENT_REPLY,
            "jargon": JARGON_REPLY,
        }
        others = [r for name, r in pool.get(stage, []) if name != fname]
        if others:
            replies["swapped"] = rng.choice(others)
        if idx > 0:
            replies["repeat"] = turns[idx - 1].get("teacher_response", "")
        last = turns[-1]
        if stage in "bc" and _stage(last) == "e" and idx < len(turns) - 1:
            replies["premature_closure"] = last.get("ground_truth_teacher", "")
        for variant in VARIANTS:
            reply = replies.get(variant, "")
            if not reply or (variant != "system" and reply == system_reply):
                continue
            items.append(
                {
                    "key": f"{fname}:{idx}:{variant}",
                    "file": fname,
                    "turn": idx,
                    "stage": stage,
                    "variant": variant,
                    "reply": reply,
                }
            )
    return items


def sample_fingerprint(items: list[dict]) -> str:
    """Hash of the items and their text, so results are only merged or compared like for like."""
    blob = json.dumps([(i["key"], i["reply"]) for i in items], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def score_item(client, model: str, backend: str, dialogue: dict, item: dict) -> dict | None:
    """Score one item: the dialogue with the target turn's reply replaced by the control."""
    d = copy.deepcopy(dialogue)
    d["dialogue"][item["turn"]]["teacher_response"] = item["reply"]
    scores = llm_judge_eval.score_turn(client, model, d, item["turn"], backend=backend)
    if scores is None:
        return None
    out = {axis: scores[axis] for axis in AXES}
    out["total"] = scores["total"]
    return out


def _cluster_bootstrap(
    diffs: np.ndarray, clusters: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """95% interval on the mean of `diffs`, resampling whole dialogues."""
    ids = np.unique(clusters)
    sums = np.array([diffs[clusters == c].sum() for c in ids])
    counts = np.array([(clusters == c).sum() for c in ids])
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(items: list[dict], scores: dict[str, list[dict]], n_boot: int, seed: int) -> dict:
    """Per-variant means, and for each control its paired drop against the system reply.

    `scores` maps item key to the list of scores from each repeat. Comparisons
    use the first repeat; the rest feed the repeatability figure.
    """
    rng = np.random.default_rng(seed)
    first = {k: v[0] for k, v in scores.items() if v}
    by_turn: dict[tuple[str, int], dict[str, dict]] = {}
    for item in items:
        if item["key"] in first:
            by_turn.setdefault((item["file"], item["turn"]), {})[item["variant"]] = first[
                item["key"]
            ]

    variants: dict[str, dict] = {}
    for variant in VARIANTS:
        rows = [v[variant] for v in by_turn.values() if variant in v]
        if not rows:
            continue
        variants[variant] = {
            "n": len(rows),
            "total": float(np.mean([r["total"] for r in rows])),
            "axes": {a: float(np.mean([r[a] for r in rows])) for a in AXES},
        }

    controls: dict[str, dict] = {}
    for variant in ("reference", *BAD_CONTROLS):
        pairs = [(k, v) for k, v in by_turn.items() if variant in v and "system" in v]
        if not pairs:
            continue
        files = np.array([k[0] for k, _ in pairs])
        drop = np.array([v["system"]["total"] - v[variant]["total"] for _, v in pairs], dtype=float)
        lo, hi = _cluster_bootstrap(drop, files, n_boot, rng)
        entry: dict = {
            "n_pairs": len(pairs),
            "mean_drop_total": float(drop.mean()),
            "ci95": [lo, hi],
            "share_scored_below_system": float((drop > 0).mean()),
            "share_tied_with_system": float((drop == 0).mean()),
            "share_scored_above_system": float((drop < 0).mean()),
        }
        if variant in BAD_CONTROLS:
            entry["target_axes"] = {
                a: float(np.mean([v["system"][a] - v[variant][a] for _, v in pairs]))
                for a in BAD_CONTROLS[variant]
            }
            entry["detected"] = bool(drop.mean() >= MIN_DROP and lo > 0)
        controls[variant] = entry

    repeated = [v for v in scores.values() if len(v) > 1]
    repeatability = None
    if repeated:
        same = [all(r["total"] == v[0]["total"] for r in v[1:]) for v in repeated]
        spread = [max(r["total"] for r in v) - min(r["total"] for r in v) for v in repeated]
        repeatability = {
            "n_items": len(repeated),
            "share_identical_total": float(np.mean(same)),
            "mean_spread_total": float(np.mean(spread)),
        }

    bad = [c for c in BAD_CONTROLS if c in controls]
    return {
        "variants": variants,
        "controls": controls,
        "repeatability": repeatability,
        "n_bad_controls": len(bad),
        "n_bad_controls_detected": sum(controls[c]["detected"] for c in bad),
    }


def print_summary(summary: dict, meta: dict) -> None:
    print(f"\nJudge: {meta['judge_model']}  ({meta['backend']})")
    print(
        f"Items scored: {meta['n_items_scored']} of {meta['n_items']} "
        f"({meta['n_turns']} turns from {meta['n_dialogues']} dialogues)"
    )
    print("\nMean score by reply type (total out of 10):")
    header = f"  {'reply':<18} {'n':>4} {'total':>6}   " + "  ".join(f"{a[:9]:>9}" for a in AXES)
    print(header)
    for name, v in summary["variants"].items():
        axes = "  ".join(f"{v['axes'][a]:>9.2f}" for a in AXES)
        print(f"  {name:<18} {v['n']:>4} {v['total']:>6.2f}   {axes}")

    print("\nDrop against the system's own reply on the same turn (positive = scored lower):")
    print(
        f"  {'control':<18} {'n':>4} {'drop':>6} {'95% interval':>16} {'below':>6} {'tied':>6}  target axes"
    )
    for name, c in summary["controls"].items():
        interval = f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"
        target = ", ".join(f"{a} {d:+.2f}" for a, d in c.get("target_axes", {}).items())
        flag = "" if "detected" not in c else ("  detected" if c["detected"] else "  NOT DETECTED")
        print(
            f"  {name:<18} {c['n_pairs']:>4} {c['mean_drop_total']:>+6.2f} {interval:>16} "
            f"{c['share_scored_below_system']:>6.0%} {c['share_tied_with_system']:>6.0%}  {target}{flag}"
        )
    if summary["repeatability"]:
        r = summary["repeatability"]
        print(
            f"\nRepeatability: {r['share_identical_total']:.0%} of {r['n_items']} repeated items got "
            f"the same total; mean spread {r['mean_spread_total']:.2f} points."
        )
    print(
        f"\nBad controls detected: {summary['n_bad_controls_detected']} of {summary['n_bad_controls']} "
        f"(detected = mean drop of at least {MIN_DROP} point with an interval above zero; a heuristic)."
    )
    print(
        "A reference reply scoring below the system's reply is not a fault by itself: the judge "
        "is shown the reference and told it is one valid move among several."
    )


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "judge"


def run(args: argparse.Namespace) -> int:
    dialogues_dir = args.results_dir / "dialogues"
    if not dialogues_dir.is_dir():
        print(f"ERROR: {dialogues_dir} not found", file=sys.stderr)
        return 1
    items = build_items(dialogues_dir, args.per_stage, args.stages, args.seed)
    fingerprint = sample_fingerprint(items)
    turns = {(i["file"], i["turn"]) for i in items}
    counts = {v: sum(i["variant"] == v for i in items) for v in VARIANTS}
    n_calls = len(items) + (args.repeats - 1) * counts["system"]

    print(
        f"Sample: {len(turns)} turns from {len({f for f, _ in turns})} dialogues, seed {args.seed}"
    )
    print("Items per reply type: " + ", ".join(f"{v} {n}" for v, n in counts.items()))
    print(f"Judge calls: {n_calls}  (sample fingerprint {fingerprint})")
    if args.dry_run:
        shown: set[str] = set()
        print("\nOne example of each reply type:")
        for item in items:
            if item["variant"] not in shown:
                shown.add(item["variant"])
                print(
                    f"  [{item['variant']}] {item['file']} turn {item['turn']}: {item['reply'][:90]}"
                )
        return 0

    model = args.model or (
        "claude-opus-4-8" if args.backend == "claude-code" else "claude-sonnet-4-6"
    )
    out_path = args.out or args.results_dir / f"judge_validation_{_safe_name(model)}.json"
    client = None
    if args.backend == "api":
        import openai

        client = openai.Client(api_key=args.api_key or "local", base_url=args.base_url)

    # Resume: keep scores from an earlier, interrupted pass over the same sample.
    scores: dict[str, list[dict]] = {}
    if out_path.exists():
        previous = json.loads(out_path.read_text())
        if previous.get("meta", {}).get("sample_fingerprint") == fingerprint:
            scores = previous.get("scores", {})
            print(f"Resuming: {sum(len(v) for v in scores.values())} scores already in {out_path}")

    dialogues = {
        name: json.loads((dialogues_dir / name).read_text()) for name in {i["file"] for i in items}
    }
    jobs: list[dict] = []
    for item in items:
        wanted = args.repeats if item["variant"] == "system" else 1
        jobs.extend([item] * max(0, wanted - len(scores.get(item["key"], []))))

    meta = {
        "results_dir": str(args.results_dir),
        "judge_model": model,
        "backend": args.backend,
        "base_url": args.base_url if args.backend == "api" else None,
        "per_stage": args.per_stage,
        "stages": args.stages,
        "seed": args.seed,
        "repeats": args.repeats,
        "sample_fingerprint": fingerprint,
        "n_items": len(items),
        "n_turns": len(turns),
        "n_dialogues": len({f for f, _ in turns}),
    }

    def save(failed: int, elapsed: float) -> dict:
        meta.update(
            n_items_scored=sum(1 for i in items if scores.get(i["key"])),
            n_failed_calls=failed,
            wall_clock_seconds=round(elapsed, 1),
        )
        summary = summarize(items, scores, args.n_boot, args.seed)
        out_path.write_text(
            json.dumps(
                {"meta": meta, "summary": summary, "items": items, "scores": scores},
                ensure_ascii=False,
                indent=1,
            )
        )
        return summary

    started = time.time()
    failed = 0
    workers = 1 if args.backend == "claude-code" else args.workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(score_item, client, model, args.backend, dialogues[j["file"]], j): j
            for j in jobs
        }
        for n, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            result = future.result()
            if result is None:
                failed += 1
            else:
                scores.setdefault(item["key"], []).append(result)
            if n % 25 == 0 or n == len(jobs):
                save(failed, time.time() - started)
                print(f"  {n}/{len(jobs)} calls, {failed} failed, {time.time() - started:.0f}s")

    summary = save(failed, time.time() - started)
    print_summary(summary, meta)
    print(f"\nWritten to {out_path}")
    if jobs and failed / len(jobs) > MAX_FAILURE_RATE:
        print(
            f"WARNING: the judge failed on {failed} of {len(jobs)} calls. A thinking model needs "
            "reasoning switched off: the judge reply is capped at 200 tokens.",
            file=sys.stderr,
        )
        return FAILURE_EXIT_CODE
    return 0


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties given their average rank."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        mask = x == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def agreement(a: dict, b: dict) -> dict:
    """Item-level agreement between two validation files over the same sample."""
    if a["meta"]["sample_fingerprint"] != b["meta"]["sample_fingerprint"]:
        raise ValueError("the two files were not produced from the same sample")
    keys = sorted(k for k in a["scores"] if a["scores"][k] and b["scores"].get(k))
    x = np.array([a["scores"][k][0]["total"] for k in keys], dtype=float)
    y = np.array([b["scores"][k][0]["total"] for k in keys], dtype=float)
    rx, ry = _ranks(x), _ranks(y)
    constant = rx.std() == 0 or ry.std() == 0
    out = {
        "n_items": len(keys),
        "spearman_total": None if constant else float(np.corrcoef(rx, ry)[0, 1]),
        "mean_abs_diff_total": float(np.abs(x - y).mean()),
        "mean_total": [float(x.mean()), float(y.mean())],
        "share_within_one_point": float((np.abs(x - y) <= 1).mean()),
    }
    system = [i for i, k in enumerate(keys) if k.endswith(":system")]
    if len(system) > 2:
        sx, sy = _ranks(x[system]), _ranks(y[system])
        flat = sx.std() == 0 or sy.std() == 0
        out["spearman_total_system_replies_only"] = (
            None if flat else float(np.corrcoef(sx, sy)[0, 1])
        )
    return out


def compare(paths: list[Path]) -> int:
    a, b = (json.loads(p.read_text()) for p in paths)
    result = agreement(a, b)
    print(f"A: {a['meta']['judge_model']}    B: {b['meta']['judge_model']}")
    print(f"Shared items: {result['n_items']}")
    print(f"Mean total: A {result['mean_total'][0]:.2f}, B {result['mean_total'][1]:.2f}")
    print(f"Mean absolute difference: {result['mean_abs_diff_total']:.2f} points")
    print(f"Within one point of each other: {result['share_within_one_point']:.0%}")
    print(f"Spearman correlation, all items: {result['spearman_total']}")
    print(
        "Spearman correlation, system replies only: "
        f"{result.get('spearman_total_system_replies_only')}"
    )
    print(
        "The all-items figure is inflated by the controls, which both judges should rank low. "
        "The system-replies figure is the one that says whether they rank real replies alike."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether an LLM judge separates good Socratic replies from bad ones."
    )
    parser.add_argument("results_dir", type=Path, nargs="?", help="Run directory with dialogues/")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    parser.add_argument("--backend", choices=["api", "claude-code"], default="api")
    parser.add_argument("--model", default=None, help="Judge model id")
    parser.add_argument("--base-url", default="https://api.anthropic.com/v1/")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent calls (api backend)")
    parser.add_argument("--per-stage", type=int, default=30, help="Turns sampled per stage")
    parser.add_argument("--stages", default="bcde", help="Stages to sample; 'a' is the opener")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Times to score each system reply; 2 or more measures repeatability",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Show the sample; judge nothing")
    args = parser.parse_args()

    if args.compare:
        sys.exit(compare(args.compare))
    if args.results_dir is None:
        parser.error("results_dir is required unless --compare is given")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
