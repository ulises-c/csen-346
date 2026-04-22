# Qwen3.5-9B Evaluation Fix Plan

**Branch:** `worktree-qwen-eval-improvement`
**Created:** 2026-04-21 · **Updated:** 2026-04-22 (revised after full WAVE HPC run)

---

## Observed vs Expected

| Source | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | State acc (overall) |
|--------|---------|---------|---------|--------|---------------------|
| Paper Table 1 (SocratTeachLLM + GPT-4o, GT consultant) | 57.4 | 33.63 | 50.77 | 41.96 | — |
| Our GPT-4o baseline (live consultant, n=681) | 44.61 | 26.04 | 38.02 | 19.60 | 25.94% |
| L40S · Qwen3.5-9B (n=25 preliminary) | 43.49 | 24.18 | 36.20 | 17.76 | 18.92% |
| **WAVE HPC · Qwen3.5-9B (n=681 full run)** | **43.72** | **24.87** | **36.76** | **18.63** | **18.93%** |

The **~13 ROUGE-1 gap to the paper** is methodological: the paper feeds SocratTeachLLM
the ground-truth consultant evaluation + action at every turn; our pipeline uses live
predictions. When the consultant predicts the wrong state, the wrong action flows to
SocratTeachLLM, degrading teacher output quality.

The **~7-point state accuracy gap vs GPT-4o** is confirmed at full scale:

| Stage | GPT-4o (n=681) | Qwen n=25 | Qwen n=681 | Root cause |
|-------|---------------|-----------|------------|-----------|
| a | 95.15% | 60.00% | **57.12%** | Misses `a0→a1` student-question trigger |
| b | 36.93% | 25.00% | **26.45%** | Weaker within-stage discrimination |
| c | 4.70%  | 10.64% | **6.99%**  | Both models fail; 22 states is hard |
| d | 5.04%  | 0.00%  | **3.43%**  | In-range at full scale (was n=25 noise) |
| e | 11.92% | 4.35%  | **14.11%** | Exceeds GPT-4o at full scale (n=25 noise) |

---

## What changed from n=25 → n=681

The L40S preliminary (n=25) was accurate on headline numbers — overall state accuracy
was 18.92% vs 18.93% at full scale — but was misleading on two specific stages:

**Stage c (10.64% → 6.99%):** The preliminary was optimistic. With only 47 c-stage
turns across 25 dialogues, a few lucky matches inflated the number. At 1,447 c-stage
turns (n=681) the true rate is 6.99% — closer to GPT-4o's 4.70% and consistent with
the inherent difficulty of 22-state classification.

**Stages d and e (d: 0% → 3.43%, e: 4.35% → 14.11%):** Both were small-sample
artifacts. With only 25 d-turns and 25 e-turns in the preliminary, a single bad cluster
of dialogues could wipe them out. At n=681 both are within range of GPT-4o: d at 3.43%
vs 5.04%, and e at 14.11% *exceeding* 11.92%. The d/e failure modes were noise, not
a real problem.

**Implication for the plan:** Action 3 (stage d/e alignment analysis and
`max_teaching_rounds` tuning) was written to explain why d=0%. That question is
answered — it was n=25 noise. Action 3 is dropped entirely.

**Stage a is the only confirmed systematic failure.** At 57.12% across 681 dialogues
it is consistent, large, and not a sampling artifact.

---

## Action 1 — Ground-truth consultant evaluation mode

**Goal:** Reproduce the paper's Table 1 setup to separate teacher quality from
consultant quality and establish a meaningful ceiling.

**What:** Add `run_single_dialogue_gt_consultant` to `src/project/kele.py`. Replays
each dialogue using the ground-truth `state` and `evaluation` stored in
`SocratDataset.json` instead of calling the live consultant. The live SocratTeachLLM
teacher is still called, so teacher quality is measured faithfully.

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

> Note: Check whether `SocratDataset.json` turn entries include an `"evaluation"` key.
> If missing, pass an empty string; the teacher prompt degrades gracefully to action-only.

**Success criterion:** GT-consultant ROUGE-1 ≥ 50 on a 25-dialogue sample.

---

## Action 2 — Diagnose and fix stage-a failures

Stage a is the only confirmed systematic failure: **57.12%** at n=681, consistent with
60.00% at n=25. All 681 a-stage ground-truth turns are `a1` (student always opens with
a concrete question). Qwen3.5-9B misses this ~43% of the time.

### 2a — Diagnostic script

**File:** `src/project/debug.py` (new)

Runs the consultant on the first N dialogues, logs raw JSON responses per turn, and
reports the JSON parse failure rate. Invoked via `python -m src.project.debug`.

```python
"""
Debug: print raw consultant predictions vs ground truth, and JSON failure rate.
Usage: python -m src.project.debug --n 5 --experiment wave
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

Run against `wave` (Qwen3.5-9B) to determine:
1. Whether Qwen outputs `a0` on turns where the student clearly asked a question
2. Whether JSON parse failures are a contributing factor (>5% is significant)

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

### 2c — Stage-a prompt hardening

If the diagnostic confirms Qwen emitting `a0` when a question is clearly present,
tighten the a-stage trigger in the consultant system prompt
(`socratic_teaching_system.py`, `阶段a：学生提问` section):

Current text:
```
a1：学生提出问题
```

Proposed replacement:
```
a1：学生提出了具体的问题或题目（包括填空题 "...是( )"、选择题、
    问答题，或任何包含具体题目描述的输入）
```

Add after the table:
```
**判断规则（重要）**
- 如果学生的输入中包含任何具体题目描述，例如"帮我解答..."、"这道题..."、
  "...是( )"、"以下哪个..."，则必须判断为 a1，不得停留在 a0。
- 只有当学生完全未提及任何题目内容时，才保持 a0。
```

**Success criterion:** Stage-a accuracy ≥ 85% on a fresh 25-dialogue eval after the
prompt change.

---

## Execution order

| Step | Task | Expected effort | Blocker? |
|------|------|-----------------|---------|
| 1 | GT-consultant eval mode + `--gt-consultant` flag | 2–3 h | No |
| 2 | `src/project/debug.py` diagnostic | 1 h | Needs Qwen vLLM running |
| 3 | JSON regex fallback (if failure rate >5%) | 30 min | After step 2 |
| 4 | Stage-a prompt hardening | 1 h | After step 2 confirms root cause |

---

## Files to create / modify

| File | Action |
|------|--------|
| `src/project/kele.py` | Add `run_single_dialogue_gt_consultant`; add `--gt-consultant` flag |
| `src/project/socratic_teaching_system.py` | Stage-a prompt hardening (2c); JSON regex fallback (2b) |
| `src/project/debug.py` | New diagnostic module |

---

## Success criteria

| Metric | Baseline (n=681) | Target |
|--------|-----------------|--------|
| GT-consultant ROUGE-1 | — (not yet run) | ≥ 50 |
| Stage-a accuracy (Qwen) | 57.12% | ≥ 85% |
| JSON parse failure rate (Qwen) | unknown | < 1% |
| Overall state accuracy (Qwen) | 18.93% | > 22% |
