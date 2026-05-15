# 8-hour autonomous run log — 2026-05-15 (overnight)

Branch: `mk/8h-autonomous-extensions` (off `claude/blissful-poitras-024c46` = PR #50 + my §4.7 fix)
Author: Claude Opus 4.7 (1M context)
Operator: Max (asleep)
Hardware: RTX 5090 32 GB, llama.cpp via uv
Window: starts 2026-05-14 23:53 PDT, target 2026-05-15 08:00 PDT

## Mission

Extend the gradient analysis and address the ROUGE recovery item in `acl_latex.tex` §4.7
while Max is asleep. Use the idle 5090 to fill the data gaps that the post-merge
investigation surfaced.

## Phase 0 — Paper correction (commit `a225b82`)

**Found:** the Gemma row in Table 7 (`tab:thinkbenefit`) treated the n=25 mini result
(41.89%) as a "think" datapoint and the n=50 tournament (35.86%) as "no-think", but
inspection of `scripts/eval_gemma4_31b.sh` + the Gemma 31B mini server log showed
no `<think>` emission — they were both default-mode runs at different sample sizes.
The claimed "+0–6 pts gradient" was sample variance, not a think-mode delta.

**Fix:** dropped the Gemma row from Table 7; restated the gradient claim as
Qwen-family-only (A3B +19, 27B +16–18); updated Abstract, Conclusion, Takeaways,
Next Steps, and Limitations to match. Abstract now 186/200 words. Patch committed
+ pushed before the autonomous run kicked off so the corrected paper is what
lands when PR #50 merges to main.

## Phase 1 — Plumbing (commit `b49a9a0`)

Added the missing eval/serve scripts:
- `scripts/eval_gemma4_26b_a4b.sh` + `configs/gemma4-26b-a4b-local.env`
- `scripts/eval_qwen35b_a3b_n50.sh` (adds `n50` mode to A3B eval)
- `src/project/socratic_teaching_unified.py`: opt-in 3-shot teacher exemplars
  gated on `KELE_FEW_SHOT_TEACHER=1` (default off → byte-identical prompt)

Added the Qwopus plumbing (commit `68f7260`):
- `scripts/serve_qwopus35b_a3b_think.sh` (no `--reasoning off`)
- `scripts/eval_qwopus35b_a3b.sh`
- `configs/qwopus35b-a3b-local.env`

Also installed `uv` into the project venv (`.venv/bin/uv`) — was missing from PATH.

## Phase 2 — Runs

### Gemma 4 26B-A4B characterization

| Run | n | State acc | ROUGE-1 | BLEU-4 | Notes |
|---|---:|---:|---:|---:|---|
| Smoke (`gemma4-26b-a4b-local-smoke-unified`) | 5 dialogues / 32 turns | **37.50%** | 33.45 | 6.08 | Default-mode (Gemma has no separable think flag); per-turn `thinking_content` captured = Gemma actually does emit reasoning, just not switchable |
| Mini | _pending_ | _pending_ | _pending_ | _pending_ | |
| Tournament reference (R9700, Ulises) | 50 | 38.67% | 32.2 | 5.7 | matches smoke within 1.2 pts |

**Per-stage smoke:** a=100.0 / b=33.33 / c=23.08 / d=20.0 / e=33.33.
Stage c (the 22-state hard middle) is unusually high vs the tournament's c=14.6.

### Other runs

_pending — A3B n=50, A3B prompt-eng, 27B Q5 mini, Qwopus_

## Open issues / surprises

1. **Gemma 4 does emit reasoning** — confirmed by per-turn `thinking_content`
   capture during the A4B smoke. The paper text says Gemma "lacks a separable
   thinking mode at our serving layer" which is still correct: the reasoning
   is always-on (no `enable_thinking=False` knob). But it isn't "no thinking at
   all" — Gemma's default mode includes substantial reasoning, which is exactly
   why the smoke/mini/tournament numbers are all comparable.

2. **Three bugs in my own eval-script template** caught at runtime:
   - Sed didn't replace `serve_gemma4_31b_q5.sh` reference (loaded 31B weights
     by mistake; killed and re-ran)
   - `uv` not on PATH — fixed by installing into project venv
   - n=5 smoke for A4B took ~7 min wall-clock — slower than expected for an MoE,
     likely due to thinking content emission

## Branch + push status

All commits pushed to `origin/mk/8h-autonomous-extensions` for morning review.
PR #50 (claude/blissful-poitras-024c46) is unchanged from Ulises's last push;
this branch is a parallel investigation, NOT auto-merged. Max decides whether
to merge this branch into PR #50 (preferred), open a new PR, or cherry-pick the
paper fix only.
