# Qwen3.5-9B Evaluation Fix Plan

**Branch:** `worktree-qwen-eval-improvement`
**Date:** 2026-04-21
**Context:** Root-cause analysis of the L40S preliminary run (n=25, Qwen3.5-9B consultant)
identified three actionable causes for the low state accuracy (18.92% overall). This plan
addresses them in priority order.

---

## Observed vs Expected

| Source | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | State acc (overall) |
|--------|---------|---------|---------|--------|---------------------|
| Paper Table 1 (SocratTeachLLM + GPT-4o, GT consultant) | 57.4 | 33.63 | 50.77 | 41.96 | — |
| Our GPT-4o baseline (live consultant, n=681) | 44.61 | 26.04 | 38.02 | 19.60 | 25.94% |
| Our L40S run (Qwen3.5-9B, n=25) | 43.49 | 24.18 | 36.20 | 17.76 | 18.92% |

The **~13 ROUGE-1 gap to the paper** is not Qwen-specific — even our GPT-4o baseline
misses by ~13 points. The gap is methodological: the paper feeds SocratTeachLLM the
ground-truth consultant evaluation + action at every turn, while our pipeline uses live
consultant predictions. When the consultant predicts the wrong state, the wrong action
flows to SocratTeachLLM, degrading teacher output quality.

The **~6-point gap between Qwen and our own GPT-4o baseline** is real and comes from:

| Stage | GPT-4o | Qwen | Root cause |
|-------|--------|------|-----------|
| a | 95.15% | 60.00% | Qwen misses the a0→a1 student-question trigger |
| b | 36.93% | 25.00% | Weaker within-stage discrimination |
| c | 4.70%  | 10.64% | Both fail; Qwen noise from defaulting to common states |
| d | 5.04%  | 0.00%  | All upstream errors compound here |
| e | 11.92% | 4.35%  | Downstream of d failure |

---

## Action 1 — Ground-truth consultant evaluation mode

**Goal:** Reproduce the paper's Table 1 setup so we have a fair apples-to-apples
comparison. This also cleanly separates *teacher quality* from *consultant quality*.

**What:** Add `run_single_dialogue_gt_consultant` to `src/project/kele.py`. This mode
replays a dialogue using the ground-truth `state` and `evaluation` stored in
`SocratDataset.json` (each turn already carries both) instead of calling the live
consultant. The live SocratTeachLLM teacher is still called, so teacher quality is
measured faithfully.

**Files:**
- `src/project/kele.py` — add the new function + `--gt-consultant` flag on the
  `evaluate` subcommand

**Sketch:**
```python
def run_single_dialogue_gt_consultant(
    system: SocraticTeachingSystem, item: dict
) -> dict:
    """Replay dialogue with ground-truth consultant outputs (matches paper eval setup)."""
    system.reset_session()
    generated_turns = []
    for turn in item["dialogue"]:
        student_input = turn["student"]
        gt_state = turn["state"]
        gt_evaluation = turn.get("evaluation", "")
        action = system.get_action_for_state(gt_state)

        system.add_to_history("student", student_input)
        system.add_to_consultant_history(gt_evaluation, gt_state, action)
        system.current_state = gt_state

        teacher_response = system.socrates_teacher(student_input, gt_evaluation, action)
        system.add_to_history("teacher", teacher_response)

        generated_turns.append({
            "student": student_input,
            "state": gt_state,
            "teacher_response": teacher_response,
            "ground_truth_teacher": turn["teacher"],
            "ground_truth_state": gt_state,
        })
        if gt_state == "e34":
            break

    return {
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "num_turns_ground_truth": len(item["dialogue"]),
        "num_turns_generated": len(generated_turns),
        "dialogue": generated_turns,
    }
```

**Success criterion:** GT-consultant ROUGE-1 ≥ 50 on a 25-dialogue sample
(within ~15% of the paper's 57.4, accounting for tokenization differences).

> Note: The SocratDataset entries do not always include an `"evaluation"` key at the
> turn level — check this before running. If missing, pass an empty string and fall
> back to the action description.

---

## Action 2 — Diagnose stage-a failures and JSON reliability

Stage a accuracy fell from 95% (GPT-4o) to 60% (Qwen3.5-9B). All 681 a-stage ground-
truth turns are `a1` (student always opens with a concrete question). Qwen3.5-9B misses
this ~40% of the time.

### 2a — Diagnostic script

**File:** `src/project/debug.py` (new)

Runs the consultant on the first N dialogues, logs raw JSON responses per turn, and
reports the JSON parse failure rate. Invoked via `python -m src.project.debug`.

```python
"""
Debug: print raw consultant predictions vs ground truth, and JSON failure rate.
Usage: python -m src.project.debug --n 5 --experiment l40s
"""
import argparse
from src.project.kele import create_system, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    system = create_system(debug=False, experiment=args.experiment)
    dataset = load_dataset(split="test")[: args.n]
    json_failures = 0
    total = 0

    for item in dataset:
        print(f"\n=== Dialogue {item['id']} ===")
        system.reset_session()
        for turn in item["dialogue"]:
            total += 1
            result = system.socratic_teaching_consultant(turn["student"])
            pred = result.get("state", "MISSING")
            gt = turn["state"]
            ok = "✓" if pred == gt else "✗"
            failed_json = "无法评估" in result.get("evaluation", "")
            if failed_json:
                json_failures += 1
            print(
                f"  {ok}  gt={gt:4s}  pred={pred:4s}  json_ok={not failed_json}"
                f"  | {turn['student'][:60]}"
            )
            if system.current_state == "e34":
                break

    rate = json_failures / total * 100 if total else 0
    print(f"\nJSON parse failures: {json_failures}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
```

Run against both `baseline` (GPT-4o) and `l40s` (Qwen3.5-9B) to quantify:
1. Where exactly Qwen outputs `a0` vs `a1`
2. Whether JSON parse failures are a factor (>5% is significant)

### 2b — JSON regex fallback

If JSON parse failures are >5%, add a regex extraction fallback in
`src/project/socratic_teaching_system.py` after the markdown-strip block
(around line 340):

```python
if not raw_content.strip().startswith("{"):
    import re
    m = re.search(r'\{[^{}]*"state"[^{}]*\}', raw_content, re.DOTALL)
    if m:
        raw_content = m.group(0)
```

This handles cases where vLLM's `json_object` enforcement fails and Qwen wraps
the JSON in prose or extra whitespace.

### 2c — Stage-a prompt hardening

If the diagnostic shows Qwen emitting `a0` when the student's input clearly contains
a question (e.g. fill-in-the-blank "...是( )" or multiple-choice text), tighten the
a-stage trigger in the consultant system prompt
(`socratic_teaching_system.py`, the `阶段a：学生提问` section):

Current text (line ~205):
```
a1：学生提出问题
```

Proposed replacement:
```
a1：学生提出了具体的问题或题目（包括填空题 "...是( )"、选择题、
    问答题，或任何包含具体题目描述的输入）
```

Add a note after the table:
```
**判断规则（重要）**
- 如果学生的输入中包含任何具体题目描述，例如"帮我解答..."、"这道题..."、
  "...是( )"、"以下哪个..."，则必须判断为 a1，不得停留在 a0。
- 只有当学生完全未提及任何题目内容时，才保持 a0。
```

**Success criterion:** Stage-a accuracy ≥ 85% on the 25-item L40S sample after
prompt change.

---

## Action 3 — Stage d/e alignment analysis

Stage d accuracy is 0% for Qwen. The ground truth has d33 at 87.7% of d-stage turns
(644 / 734 total). The question is whether Qwen never emits d33, or emits it at the
wrong turns due to upstream state drift.

### 3a — Stage alignment metric

Add `compute_stage_alignment` to `src/project/metrics.py` to count generated vs GT
turns per stage per dialogue:

```python
def compute_stage_alignment(dialogues_dir: Path) -> list[dict]:
    """Per-dialogue stage-count comparison: generated vs ground truth."""
    results = []
    for f in sorted(dialogues_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if "error" in data:
            continue
        gt   = [t["ground_truth_state"][0] for t in data["dialogue"]]
        pred = [t["state"][0]              for t in data["dialogue"]]
        results.append({
            "id": data["id"],
            "gt_stage_counts":   {s: gt.count(s)   for s in "abcde"},
            "pred_stage_counts": {s: pred.count(s) for s in "abcde"},
        })
    return results
```

Expose it via `python -m src.project.metrics --alignment <dialogues_dir>` or a small
`scripts/alignment_report.py` wrapper. Compare the L40S run against the GPT-4o
baseline run to see if Qwen collapses into c-stage and never advances to d.

### 3b — max_teaching_rounds tuning (if 3a shows force-to-d33 misalignment)

`process_student_input` (lines 465–486 of `socratic_teaching_system.py`) forces the
state to d33 when `teaching_rounds >= max_teaching_rounds`. If Qwen3.5-9B's dialogues
consistently reach the limit at different turns than the ground truth, the force-to-d33
fires at the wrong position, making d-accuracy zero even when Qwen would eventually
reach d33 on its own.

Fix options (evaluate after seeing 3a data):
- Increase `MAX_TEACHING_ROUNDS` from 8 to 10 in `configs/l40s.env` to give Qwen more
  room before forcing.
- Add a guard: only force-to-d33 after at least 2 c-stage turns have been recorded
  (prevents premature forcing when Qwen skips c-stage).

---

## Execution order

| Step | Task | Expected effort | Blocker? |
|------|------|-----------------|---------|
| 1 | `run_single_dialogue_gt_consultant` + `--gt-consultant` flag | 2–3 h | No — needs SocratDataset eval key check first |
| 2 | `src/project/debug.py` | 1 h | Needs L40S vLLM running |
| 3 | JSON regex fallback (if failure rate >5%) | 30 min | After step 2 |
| 4 | Stage-a prompt hardening | 1 h | After step 2 confirms root cause |
| 5 | `compute_stage_alignment` metric + report | 1 h | No |
| 6 | `max_teaching_rounds` tuning | 1 h | After step 5 |

---

## Files to create / modify

| File | Action |
|------|--------|
| `src/project/kele.py` | Add `run_single_dialogue_gt_consultant`; add `--gt-consultant` flag to `evaluate` subcommand |
| `src/project/socratic_teaching_system.py` | Harden stage-a prompt (2c); add JSON regex fallback (2b) |
| `src/project/metrics.py` | Add `compute_stage_alignment` |
| `src/project/debug.py` | New diagnostic module (`python -m src.project.debug`) |
| `configs/l40s.env` | Possibly increase `MAX_TEACHING_ROUNDS` (after step 5 analysis) |

---

## Success criteria

| Metric | Current (L40S n=25) | Target |
|--------|---------------------|--------|
| GT-consultant ROUGE-1 | — | ≥ 50 |
| Stage-a accuracy (Qwen, L40S) | 60% | ≥ 85% |
| JSON parse failure rate (Qwen, L40S) | unknown | < 1% |
| Stage-d accuracy (Qwen, full run) | 0% | > 0% |
| Overall state accuracy (Qwen, full run) | 18.92% | > 22% |
