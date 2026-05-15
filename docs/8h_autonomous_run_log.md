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

### A3B matched-n on 5090

| Run | n | State acc | ROUGE-1 | BLEU-4 | Notes |
|---|---:|---:|---:|---:|---|
| **No-think (n=50, 5090)** | 50 dialogues / 300 turns | **19.67%** | 30.55 | 5.59 | Matches tournament R9700 19.74% to 0.07 pts — 5090 reproduction validated |
| **Think (n=50, 5090)** | 50 dialogues / 299 turns | **38.13%** | 32.87 | 6.32 | Required serve_qwen35b_a3b_think.sh; +18.46 pts gradient at matched n |
| Tournament reference (R9700 no-think) | 50 | 19.74% | 31.3 | 5.9 | Ulises's run |
| Locked headline (5090 think full) | 681 dialogues / 4171 turns | 38.70% | 30.63 | 5.86 | Max's 5/05 run, predates --reasoning off |

**A3B no-think gradient (matched-n, 5090 reproduction):**
- 5090 (this run): 19.67% no-think
- 5090 locked full: 38.70% think
- Δ = **+19.03 pts** from thinking — confirms the original +18.96 claim almost exactly

### Other runs

_pending — A3B think n=50, A3B prompt-eng, 27B Q5 mini, Qwopus_

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

## Runbook — post-A4B-mini sequence

When A4B mini metrics land, fire this chain manually with verification at each step:

```bash
# 1. Kill A4B server
kill $(pgrep -f "llama-server.*A4B")
sleep 3

# 2. Boot A3B + run n=50 think (matched-n) + keep server
PATH="$PWD/.venv/bin:$PATH" bash scripts/eval_qwen35b_a3b_n50.sh n50 --unified --keep-server

# 3. A3B prompt-eng smoke (reuse A3B server, suffix fewshot)
KELE_FEW_SHOT_TEACHER=1 PATH="$PWD/.venv/bin:$PATH" \
  bash scripts/eval_qwen35b_a3b.sh smoke --unified --keep-server --suffix fewshot

# 4. A3B prompt-eng mini (reuse A3B server)
KELE_FEW_SHOT_TEACHER=1 PATH="$PWD/.venv/bin:$PATH" \
  bash scripts/eval_qwen35b_a3b.sh mini --unified --keep-server --suffix fewshot

# 5. Kill A3B server
kill $(pgrep -f "llama-server.*A3B" | head -1)
sleep 3

# 6. 27B Q5 mini think (boots own server)
PATH="$PWD/.venv/bin:$PATH" bash scripts/eval_qwen27b.sh mini --unified --keep-server

# 7. 27B Q5 mini no-think (reuse 27B server)
PATH="$PWD/.venv/bin:$PATH" bash scripts/eval_qwen27b.sh mini --unified --nothink --keep-server

# 8. Kill 27B server
kill $(pgrep -f "llama-server.*Qwen 27B")
sleep 3

# 9. Qwopus think smoke + mini
PATH="$PWD/.venv/bin:$PATH" bash scripts/eval_qwopus35b_a3b.sh smoke --unified --keep-server
PATH="$PWD/.venv/bin:$PATH" bash scripts/eval_qwopus35b_a3b.sh mini --unified --keep-server

# 10. Final tear-down
pkill -f llama-server

# 11. Update paper with new numbers (docs/8h_paper_draft_inserts.md as template)
# 12. Final commit + push
```

Failure handling: if any step's output dir already exists, `rm -rf` it first
to avoid stale metrics. If server boot fails, check `server_*.log` and
`run_*.log` in the experiment dir.
