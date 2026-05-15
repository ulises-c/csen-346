# Morning briefing — 8h autonomous run results

**Date:** 2026-05-15
**Branch:** `mk/8h-autonomous-extensions` (off PR #50's `claude/blissful-poitras-024c46`)
**Window:** 23:53 PDT → ~07:00 PDT
**Operator:** Claude Opus 4.7 (1M ctx), Max asleep
**Hardware:** RTX 5090 32 GB, llama.cpp via uv (installed into project venv)

## TL;DR for the morning

I extended the paper's §4.7 with 10 new datapoints and corrected a flawed claim before PR #50 merges. The headline result of the run is that **3-shot teacher exemplars on the locked A3B configuration are a Pareto improvement at mini scale: +8.07 pts state accuracy AND +2.98 pts ROUGE-1.** This bumps Next Steps item 2 from "queued LoRA experiment" to "validated prompt-eng wins; LoRA optional."

## What changed in the paper (`acl_latex.tex`)

1. **Patched the bad Gemma row** in Table 7 (gradient table) — the prior commit treated 41.89% (n=25 mini) as "think" and 35.86% (n=50 tournament) as "no-think" for Gemma 4 31B, but verification showed Gemma 4 doesn't have a separable thinking toggle. The "+0–6 pts" was sample variance, not a gradient. Replaced with Qwen-family-only gradient claim.

2. **Table 7 now has 5 rows** (was 2): A3B (full + matched-n), Qwopus, 27B Q5 (smoke + mini). Gradient robust across n=25, n=33, n=50, n=681.

3. **NEW §4.7.2 "Surface-form recovery via 3-shot teacher exemplars"** + new Table 8. Shows the Pareto win at mini scale and explains the smoke regression as small-n outlier.

4. **NEW paragraph on A4B 5090 characterization** — smoke 37.50% / mini 38.78% (matches R9700 tournament 38.67% within 0.6 pts). A4B is the cost-efficient fallback, NOT the new headline.

5. **Abstract** (193/200 words) — added one-clause mention of ROUGE recovery.

6. **Conclusion + Next Steps + Limitations** — updated to reflect validated state.

## Branch state

19+ commits on `mk/8h-autonomous-extensions`, all pushed. PR #50 unchanged from Ulises's last push — this is a parallel investigation branch. Decision for you:

- **Option A (preferred):** merge `mk/8h-autonomous-extensions` into `claude/blissful-poitras-024c46`, then PR #50 lands the whole bundle.
- **Option B:** open a new PR from `mk/8h-autonomous-extensions` to main (after PR #50 lands).
- **Option C:** cherry-pick just the paper fix (commit a225b82) if you want to land that surgically.

## New scripts + plumbing added

| File | Purpose |
|---|---|
| `scripts/eval_gemma4_26b_a4b.sh` + `configs/gemma4-26b-a4b-local.env` | A4B eval orchestrator |
| `scripts/eval_qwen35b_a3b_n50.sh` + `scripts/eval_qwen35b_a3b_n50_think.sh` | A3B n=50 mode (think variant explicitly enables Qwen3 thinking) |
| `scripts/serve_qwen35b_a3b_think.sh` | A3B serve script WITHOUT `--reasoning off` (the regular one disables thinking server-side since the 5/11 tournament work) |
| `scripts/serve_qwen27b_q5_think.sh` + `scripts/eval_qwen27b_think.sh` | Same pattern for 27B Q5 |
| `scripts/eval_qwopus35b_a3b.sh` + `scripts/serve_qwopus35b_a3b_think.sh` + `configs/qwopus35b-a3b-local.env` | Qwopus with thinking enabled |
| `scripts/summarize_8h_results.py` | Consolidates all metrics into a markdown table |
| `src/project/socratic_teaching_unified.py` | Added opt-in `KELE_FEW_SHOT_TEACHER=1` env var that injects 3-shot teacher exemplars into the unified prompt |

## Gotchas I hit and fixed (in case anything fails for you in the morning)

1. **`uv` not on PATH** — installed into `.venv/bin/uv`. All eval invocations need `PATH="$PWD/.venv/bin:$PATH"` prefix.

2. **`--reasoning off` baked in** — `scripts/serve_qwen35b_a3b.sh` and `scripts/serve_qwen27b_q5.sh` both have `--reasoning off` since commit ae9fd69 (5/11). For think-mode work, use the `_think.sh` variants.

3. **A4B serve chain bug** — my first sed-copy of the 31B eval missed `serve_gemma4_31b_q5.sh`, which loaded the 31B weights with an A4B alias. Caught immediately by the alias check; fixed before consuming GPU time.

4. **Qwopus weight path** — original `scripts/serve_qwopus35b_a3b.sh` points to `~/models/Qwopus...` (Ulises's R9700 path). Downloaded weights to `~/Documents/models/weights/` (5090 convention) and pointed the new `_think.sh` variant there.

## Open questions for the morning

1. **Is the ROUGE recovery mini result ready to publish?** I claim yes at mini ($n{=}148$ turns). Full-run confirmation ($n{=}681$) is the obvious next experiment but would cost ~16h.

2. **Does the 27B Q5 mini gradient (+10.52) replace the smoke gradient (+16.58) in the paper, or do we keep both?** The current Table 7 shows both — wider range is honest but less punchy.

3. **A3B + 3-shot at n=50 in progress** (kicked off ~05:05, ETA 06:00). If it lands by 8 AM, the prompt-eng result will have 3 sample sizes (n=5, n=25, n=50). If not, mini is the load-bearing measurement.

## All eval results (final state)

Run `uv run python scripts/summarize_8h_results.py` for the current consolidated table.
