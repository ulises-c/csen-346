"""LLM-judge evaluation over saved dialogue logs (absolute Socratic quality).

Runs Claude as a single judge with a 4-axis rubric:
1. Socratic validity (0-3): Valid teaching move at the current stage?
2. Advancement (0-3): Does the response advance student reasoning?
3. Age-appropriateness (0-2): Vocabulary suitable for elementary?
4. Question-form fidelity (0-2): Single question, not over-leading.

Total per turn: 0-10. Aggregated per dialogue + per config + per Socratic stage.
This is an *absolute*-quality read, independent of ROUGE/BLEU overlap-with-reference
(the analysis-plan LLM-judge task; confirms the oracle-arm SFT-vs-base finding).

Two backends:
- claude-code (DEFAULT): judges via the Claude Code CLI on the logged-in Anthropic
  SUBSCRIPTION — no API key. Sequential (parallel `claude -p` would race one account).
  Default model claude-opus-4-8. ANTHROPIC_API_KEY is stripped from the child env so
  it can't silently fall back to API billing.
- api: OpenAI-compat endpoint with ANTHROPIC_API_KEY (default model claude-sonnet-4-6),
  parallel via --workers.

Usage:
  # subscription smoke (5 turns) then full arm:
  uv run python scripts/llm_judge_eval.py results/gemma4-12b-sft-oracle --max-turns 5
  uv run python scripts/llm_judge_eval.py results/gemma4-12b-sft-oracle
  # api backend:
  uv run python scripts/llm_judge_eval.py <dir> --backend api --workers 10

Output: <results_dir>/<out-name> (default judge_summary.json) with per-turn +
per-dialogue + per-stage + overall scores.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

ALL_STATES = (
    ["a0", "a1"]
    + [f"b{i}" for i in range(2, 8)]
    + [f"c{i}" for i in range(8, 30)]
    + [f"d{i}" for i in range(30, 34)]
    + ["e34"]
)

JUDGE_SYSTEM_PROMPT = """You are an expert pedagogy evaluator scoring an AI teacher's Socratic response to an elementary school student. You assess responses on four axes, given the conversation history and the ground-truth pedagogical state.

The SocRule framework has 5 stages:
- a (questioning): student poses a question
- b (concept probing): probe student concepts
- c (inductive reasoning): help student reason through misconceptions
- d (rule construction): help student build correct rule and apply it
- e (closure): summarize and confirm understanding

You score on four axes. Respond ONLY with a JSON object:

{
  "socratic_validity": 0-3,    // Is this a valid Socratic teaching move at the current stage? 3=textbook, 2=acceptable, 1=marginal, 0=wrong stage/move
  "advancement": 0-3,           // Does this response advance the student's reasoning toward the answer? 3=clearly advances, 2=neutral but valid, 1=stalls, 0=confuses/regresses
  "age_appropriateness": 0-2,   // Is vocabulary and complexity right for elementary school? 2=well-calibrated, 1=borderline, 0=too complex or too simple
  "question_form": 0-2,         // Is the response a single Socratic question? 2=clean single question, 1=multiple questions or hint, 0=statement only or over-leading
  "comment": "<brief one-line rationale>"
}

Total possible: 10 points (3+3+2+2). Be calibrated — a perfect 10 should be rare.
"""


def build_judge_user_prompt(dialogue: dict, turn_idx: int) -> str:
    """Build the per-turn judge prompt with conversation history."""
    turns = dialogue.get("dialogue", [])
    if turn_idx >= len(turns):
        return ""
    target = turns[turn_idx]

    # History (previous turns)
    history_lines = []
    for t in turns[:turn_idx]:
        s = t.get("student", "")
        tr = t.get("teacher_response", t.get("teacher", ""))
        if s:
            history_lines.append(f"STUDENT: {s}")
        if tr:
            history_lines.append(f"TEACHER: {tr}")
    history = "\n".join(history_lines) if history_lines else "(start of dialogue)"

    # Current turn
    student = target.get("student", "")
    teacher_response = target.get("teacher_response", target.get("teacher", ""))
    gt_state = target.get("ground_truth_state", target.get("state", "?"))
    gt_teacher = target.get("ground_truth_teacher", "")

    return f"""Conversation history:
{history}

Current turn:
STUDENT: {student}
GROUND-TRUTH STATE (for reference, do not penalize differences in routing): {gt_state}
GROUND-TRUTH REFERENCE RESPONSE (one valid Socratic move, but not the only valid one): {gt_teacher}

AI TEACHER RESPONSE TO SCORE: {teacher_response}

Score the AI teacher response."""


def _parse_rubric(content: str) -> dict | None:
    """Parse the judge's JSON reply into validated axis scores, or None."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        scores = json.loads(content)
    except json.JSONDecodeError:
        return None
    for k in ("socratic_validity", "advancement", "age_appropriateness", "question_form"):
        if k not in scores:
            return None
        scores[k] = int(scores[k])
    scores["total"] = (
        scores["socratic_validity"]
        + scores["advancement"]
        + scores["age_appropriateness"]
        + scores["question_form"]
    )
    return scores


def _claude_code_judge(model: str, prompt: str, max_retries: int = 5) -> dict | None:
    """Run one judge call through the Claude Code CLI on the logged-in SUBSCRIPTION.

    ANTHROPIC_API_KEY is stripped from the child env so the CLI cannot silently
    fall back to API-key billing (it takes precedence over the subscription when
    set). No tools / no MCP / no skills — a pure text-in/JSON-out judge.
    """
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        "--disable-slash-commands",
    ]
    for attempt in range(max_retries):
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
            )
            if proc.returncode != 0:
                print(f"  claude -p rc={proc.returncode}: {proc.stderr[:200]}", file=sys.stderr)
                time.sleep(min(2**attempt, 30))
                continue
            envelope = json.loads(proc.stdout)
            if envelope.get("is_error"):
                time.sleep(min(2**attempt, 30))
                continue
            usage = envelope.get("usage", {})
            # input_tokens is the UNCACHED portion only; the bulk of a repeated judge
            # prompt is prompt-cache hits, so add the cache buckets for a true count.
            usage_in = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )
            return {
                "content": envelope.get("result") or "",
                "usage_in": usage_in,
                "usage_out": usage.get("output_tokens", 0),
                "cost_usd": envelope.get("total_cost_usd", 0.0),
            }
        except subprocess.TimeoutExpired:
            print(f"  claude -p timeout (attempt {attempt + 1})", file=sys.stderr)
            time.sleep(min(2**attempt, 30))
        except json.JSONDecodeError:
            print(f"  claude -p non-JSON stdout (attempt {attempt + 1})", file=sys.stderr)
            time.sleep(min(2**attempt, 30))
    return None


def score_turn(
    client: openai.Client | None,
    model: str,
    dialogue: dict,
    turn_idx: int,
    backend: str = "api",
    max_retries: int = 6,
) -> dict | None:
    """Score a single turn via the judge. Returns dict with axis scores or None on failure."""
    user_prompt = build_judge_user_prompt(dialogue, turn_idx)
    if not user_prompt:
        return None

    if backend == "claude-code":
        r = _claude_code_judge(model, f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}", max_retries)
        if r is None:
            return None
        scores = _parse_rubric(r["content"])
        if scores is None:
            return None
        scores["usage_in"] = r["usage_in"]
        scores["usage_out"] = r["usage_out"]
        scores["cost_usd"] = r["cost_usd"]
        return scores

    assert client is not None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0,
            )
            scores = _parse_rubric(resp.choices[0].message.content or "")
            if scores is None:
                return None
            scores["usage_in"] = resp.usage.prompt_tokens if resp.usage else 0
            scores["usage_out"] = resp.usage.completion_tokens if resp.usage else 0
            return scores
        except openai.RateLimitError:
            wait = min(2**attempt, 30)
            time.sleep(wait)
        except (openai.APIStatusError, openai.APIConnectionError):
            wait = min(2**attempt, 30)
            time.sleep(wait)
        except Exception as e:
            print(f"  judge error (attempt {attempt + 1}): {e}", file=sys.stderr)
            time.sleep(2)

    return None


def judge_dialogue(
    client: openai.Client | None, model: str, dialogue: dict, backend: str = "api"
) -> list[dict]:
    """Judge every turn in a dialogue. Returns list of scored turns (skipping failures)."""
    out = []
    turns = dialogue.get("dialogue", [])
    for i in range(len(turns)):
        score = score_turn(client, model, dialogue, i, backend=backend)
        if score is not None:
            score["turn_idx"] = i
            score["ground_truth_state"] = turns[i].get("ground_truth_state", "?")
            out.append(score)
    return out


def _summarize(
    results_dir: Path,
    model: str,
    backend: str,
    all_turn_scores: list[dict],
    per_dialogue_summaries: list[dict],
    elapsed: float,
    out_name: str,
) -> dict:
    """Aggregate scored turns into the summary dict + write it to disk."""
    per_stage = {s: {"n": 0, "total": 0.0} for s in "abcde"}
    for s in all_turn_scores:
        gt = s.get("ground_truth_state", "?")
        if gt and gt[0] in per_stage:
            per_stage[gt[0]]["n"] += 1
            per_stage[gt[0]]["total"] += s["total"]
    for k in per_stage:
        n = per_stage[k]["n"]
        per_stage[k]["avg"] = per_stage[k]["total"] / n if n else 0.0

    n_turns = len(all_turn_scores)
    total_in = sum(s.get("usage_in", 0) for s in all_turn_scores)
    total_out = sum(s.get("usage_out", 0) for s in all_turn_scores)
    # claude-code backend reports per-call subscription-equivalent cost directly;
    # the api backend estimates from tokens at Sonnet 4.6 rates.
    if backend == "claude-code":
        cost = round(sum(s.get("cost_usd", 0.0) for s in all_turn_scores), 4)
    else:
        cost = round(total_in / 1e6 * 3.0 + total_out / 1e6 * 15.0, 4)

    def _axis(name: str) -> float:
        return sum(s[name] for s in all_turn_scores) / n_turns if n_turns else 0

    summary = {
        "results_dir": str(results_dir),
        "judge_model": model,
        "backend": backend,
        "n_dialogues_judged": len(per_dialogue_summaries),
        "n_turns_judged": n_turns,
        "wall_clock_seconds": elapsed,
        "overall_avg": sum(s["total"] for s in all_turn_scores) / n_turns if n_turns else 0,
        "axis_avgs": {
            "socratic_validity": _axis("socratic_validity"),
            "advancement": _axis("advancement"),
            "age_appropriateness": _axis("age_appropriateness"),
            "question_form": _axis("question_form"),
        },
        "per_stage": per_stage,
        "tokens_in": total_in,
        "tokens_out": total_out,
        "cost_usd": cost,
        "per_dialogue": per_dialogue_summaries,
    }
    summary_path = results_dir / out_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary written to {summary_path}")
    print(
        f"Overall: {summary['overall_avg']:.2f}/10  ({n_turns} turns, "
        f"${cost:.3f} cost, {elapsed:.0f}s wall clock)"
    )
    return summary


def _dialogue_summary(fname: str, did: int, scores: list[dict]) -> dict:
    dlg_n = len(scores)
    return {
        "file": fname,
        "id": did,
        "n_turns_judged": dlg_n,
        "avg_total": sum(s["total"] for s in scores) / dlg_n,
        "avg_validity": sum(s["socratic_validity"] for s in scores) / dlg_n,
        "avg_advancement": sum(s["advancement"] for s in scores) / dlg_n,
        "avg_age": sum(s["age_appropriateness"] for s in scores) / dlg_n,
        "avg_qform": sum(s["question_form"] for s in scores) / dlg_n,
    }


def judge_sequential(
    results_dir: Path,
    dialogue_files: list[Path],
    model: str,
    backend: str,
    max_turns: int | None,
    out_name: str,
) -> dict:
    """Sequential judge — one call at a time. Used for the claude-code (subscription)
    backend, where parallel `claude -p` processes would race on the same account.
    max_turns caps the total turns scored (for a cheap smoke run)."""
    print(f"\n=== Judging {len(dialogue_files)} dialogues in {results_dir} (sequential) ===")
    print(f"Judge model: {model}, backend: {backend}, max_turns: {max_turns}")
    started = time.time()
    all_turn_scores: list[dict] = []
    per_dialogue_summaries: list[dict] = []
    for dfile in dialogue_files:
        if max_turns is not None and len(all_turn_scores) >= max_turns:
            break
        d = json.loads(dfile.read_text())
        turns = d.get("dialogue", [])
        dlg_scores = []
        for i in range(len(turns)):
            if max_turns is not None and len(all_turn_scores) >= max_turns:
                break
            score = score_turn(None, model, d, i, backend=backend)
            if score is not None:
                score["turn_idx"] = i
                score["ground_truth_state"] = turns[i].get("ground_truth_state", "?")
                dlg_scores.append(score)
                all_turn_scores.append(score)
                print(
                    f"  turn {len(all_turn_scores)} [{dfile.name} #{i} "
                    f"state={score['ground_truth_state']}]: {score['total']}/10",
                    flush=True,
                )
        if dlg_scores:
            per_dialogue_summaries.append(
                _dialogue_summary(dfile.name, d.get("id", -1), dlg_scores)
            )
    return _summarize(
        results_dir,
        model,
        backend,
        all_turn_scores,
        per_dialogue_summaries,
        time.time() - started,
        out_name,
    )


def judge_results_dir(
    results_dir: Path,
    client: openai.Client | None,
    model: str,
    workers: int,
    backend: str = "api",
    max_turns: int | None = None,
    out_name: str = "judge_summary.json",
) -> dict:
    """Run judge over all dialogues in results_dir. Returns summary dict."""
    dialogues_dir = results_dir / "dialogues"
    if not dialogues_dir.is_dir():
        raise FileNotFoundError(f"No dialogues/ in {results_dir}")

    dialogue_files = sorted(dialogues_dir.glob("*.json"))

    if backend == "claude-code":
        return judge_sequential(results_dir, dialogue_files, model, backend, max_turns, out_name)

    print(f"\n=== Judging {len(dialogue_files)} dialogues in {results_dir} ===")
    print(f"Judge model: {model}, workers: {workers}")

    started = time.time()
    all_turn_scores = []
    per_dialogue_summaries = []

    def _judge_one(dfile: Path):
        d = json.loads(dfile.read_text())
        scores = judge_dialogue(client, model, d, backend=backend)
        return dfile.name, d.get("id", -1), scores

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_judge_one, f) for f in dialogue_files]
        for i, fut in enumerate(as_completed(futures)):
            fname, did, scores = fut.result()
            if scores:
                per_dialogue_summaries.append(_dialogue_summary(fname, did, scores))
                all_turn_scores.extend(scores)
            elapsed = time.time() - started
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(dialogue_files) - i - 1) / rate if rate > 0 else 0
            avg = per_dialogue_summaries[-1]["avg_total"] if scores else 0
            print(
                f"  {i + 1}/{len(dialogue_files)}  {fname}  "
                f"({len(scores) if scores else 0} turns, avg {avg:.2f}/10) "
                f"{elapsed:.0f}s elapsed, {remaining:.0f}s remaining",
                flush=True,
            )

    return _summarize(
        results_dir,
        model,
        backend,
        all_turn_scores,
        per_dialogue_summaries,
        time.time() - started,
        out_name,
    )


def build_stratified_sample(
    dialogues_dir: Path, per_stage: int, stages: str, seed: int
) -> list[tuple[str, int, str]]:
    """Pick `per_stage` turns for each Socratic stage in `stages`, seeded.

    Returns (dialogue_file, turn_idx, stage) tuples. Both oracle arms replay the
    same ground-truth student turns and hit e34 at the same turn, so a sample keyed
    on (file, turn_idx) is identical and comparable across arms — a paired design.
    """
    import random

    by_stage: dict[str, list[tuple[str, int, str]]] = {s: [] for s in stages}
    for dfile in sorted(dialogues_dir.glob("*.json")):
        d = json.loads(dfile.read_text())
        for i, turn in enumerate(d.get("dialogue", [])):
            gt = turn.get("ground_truth_state") or turn.get("state") or "?"
            stage = gt[0] if gt else "?"
            if stage in by_stage:
                by_stage[stage].append((dfile.name, i, stage))

    rng = random.Random(seed)
    sample: list[tuple[str, int, str]] = []
    for s in stages:
        pool = by_stage[s]
        if len(pool) <= per_stage:
            print(f"  stage {s}: {len(pool)} available <= {per_stage} requested — taking all")
            sample.extend(pool)
        else:
            sample.extend(rng.sample(pool, per_stage))
    return sample


def _judge_arm_on_sample(
    arm_dir: Path, sample: list[tuple[str, int, str]], model: str, backend: str
) -> list[dict]:
    """Score every sampled (file, turn_idx) turn in one arm. Sequential (subscription)."""
    dialogues_dir = arm_dir / "dialogues"
    cache: dict[str, dict] = {}
    scores: list[dict] = []
    for n, (fname, turn_idx, stage) in enumerate(sample, 1):
        if fname not in cache:
            cache[fname] = json.loads((dialogues_dir / fname).read_text())
        s = score_turn(None, model, cache[fname], turn_idx, backend=backend)
        if s is not None:
            s["turn_idx"] = turn_idx
            s["ground_truth_state"] = cache[fname]["dialogue"][turn_idx].get(
                "ground_truth_state", "?"
            )
            scores.append(s)
            print(
                f"  [{arm_dir.name}] {n}/{len(sample)} {fname}#{turn_idx} "
                f"({stage}): {s['total']}/10",
                flush=True,
            )
    return scores


def judge_paired_compare(
    arm_a: Path,
    arm_b: Path,
    model: str,
    backend: str,
    per_stage: int,
    stages: str,
    seed: int,
    out_path: Path,
) -> dict:
    """Judge both arms on ONE stratified paired sample; report per-arm scores + A−B deltas."""
    sample = build_stratified_sample(arm_a / "dialogues", per_stage, stages, seed)
    print(f"\nStratified sample: {len(sample)} turns/arm across stages '{stages}' (seed {seed})")

    started = time.time()
    arm_scores = {}
    for arm in (arm_a, arm_b):
        print(f"\n=== Judging arm: {arm.name} ({len(sample)} turns, {model}) ===")
        scores = _judge_arm_on_sample(arm, sample, model, backend)
        arm_scores[arm.name] = _summarize(
            arm, model, backend, scores, [], time.time() - started, f".judge_{arm.name}.tmp.json"
        )
        (arm / f".judge_{arm.name}.tmp.json").unlink(missing_ok=True)

    a, b = arm_scores[arm_a.name], arm_scores[arm_b.name]
    axes = ["socratic_validity", "advancement", "age_appropriateness", "question_form"]
    delta = {
        "overall": round(a["overall_avg"] - b["overall_avg"], 3),
        "axes": {ax: round(a["axis_avgs"][ax] - b["axis_avgs"][ax], 3) for ax in axes},
        "per_stage": {
            s: round(a["per_stage"][s]["avg"] - b["per_stage"][s]["avg"], 3)
            for s in stages
            if a["per_stage"][s]["n"] and b["per_stage"][s]["n"]
        },
    }
    out = {
        "judge_model": model,
        "backend": backend,
        "arm_a": arm_a.name,
        "arm_b": arm_b.name,
        "per_stage_n": per_stage,
        "stages": stages,
        "seed": seed,
        "sample_size_per_arm": len(sample),
        "wall_clock_seconds": round(time.time() - started, 1),
        "arms": {
            arm_a.name: {k: a[k] for k in ("overall_avg", "axis_avgs", "per_stage", "cost_usd")},
            arm_b.name: {k: b[k] for k in ("overall_avg", "axis_avgs", "per_stage", "cost_usd")},
        },
        "delta_a_minus_b": delta,
        "sample": sample,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nComparison written to {out_path}")
    print(
        f"{arm_a.name} {a['overall_avg']:.2f}  vs  {arm_b.name} {b['overall_avg']:.2f}  "
        f"→ Δ(a−b) = {delta['overall']:+.2f}/10"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="LLM-judge eval over saved dialogues")
    parser.add_argument("results_dir", type=Path, help="Path to results dir containing dialogues/")
    parser.add_argument(
        "--backend",
        choices=["api", "claude-code"],
        default="claude-code",
        help="'claude-code' judges via the Claude Code CLI on the logged-in "
        "SUBSCRIPTION (no API key, sequential); 'api' uses ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model ID. Default: claude-opus-4-8 (claude-code) / claude-sonnet-4-6 (api).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent judge calls (api only)")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Cap total turns scored (claude-code backend) — use for a cheap smoke run.",
    )
    parser.add_argument(
        "--out-name",
        default="judge_summary.json",
        help="Summary filename written into results_dir.",
    )
    parser.add_argument("--api-key", default=None, help="Override ANTHROPIC_API_KEY (api backend)")
    parser.add_argument(
        "--base-url",
        default="https://api.anthropic.com/v1/",
        help="OpenAI-compat endpoint base URL (api backend)",
    )
    # Paired stratified compare (the absolute-quality scenario): judge results_dir
    # (arm A) and --compare-with (arm B) on ONE stratified sample; report A−B deltas.
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Second results dir (arm B). Enables paired stratified compare mode.",
    )
    parser.add_argument(
        "--per-stage", type=int, default=40, help="Turns sampled per Socratic stage (compare mode)."
    )
    parser.add_argument(
        "--stages", default="bcde", help="Stages to sample (compare mode; 'a' is a trivial opener)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (compare mode).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/llm_judge_oracle_compare.json"),
        help="Output path for the compare summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compare mode: build + report the sample and estimated calls, judge nothing.",
    )
    args = parser.parse_args()

    if args.compare_with is not None:
        model = args.model or (
            "claude-opus-4-8" if args.backend == "claude-code" else "claude-sonnet-4-6"
        )
        if args.dry_run:
            sample = build_stratified_sample(
                args.results_dir / "dialogues", args.per_stage, args.stages, args.seed
            )
            counts: dict[str, int] = {}
            for _f, _i, s in sample:
                counts[s] = counts.get(s, 0) + 1
            print(f"\nDRY RUN — no judging. Sample per arm: {len(sample)} turns {counts}")
            print(f"Both arms → {2 * len(sample)} total {model} calls (~sequential).")
            return
        judge_paired_compare(
            args.results_dir,
            args.compare_with,
            model,
            args.backend,
            args.per_stage,
            args.stages,
            args.seed,
            args.out,
        )
        return

    if args.backend == "claude-code":
        model = args.model or "claude-opus-4-8"
        judge_results_dir(
            args.results_dir,
            None,
            model,
            args.workers,
            backend="claude-code",
            max_turns=args.max_turns,
            out_name=args.out_name,
        )
        return

    model = args.model or "claude-sonnet-4-6"
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # try .env
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = openai.Client(api_key=api_key, base_url=args.base_url)
    judge_results_dir(
        args.results_dir, client, model, args.workers, backend="api", out_name=args.out_name
    )


if __name__ == "__main__":
    main()
