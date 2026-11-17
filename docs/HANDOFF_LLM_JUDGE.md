# Handoff — LLM-judge on absolute Socratic quality (Opus 4.8 via subscription)

> **✅ SUPERSEDED — this run is DONE** (commit `90f78ec`, 2026-07-13). Result:
> `results/llm_judge_oracle_compare.json`, SFT +0.38/10 overall (confirms the oracle direction,
> corrects the magnitude — see `docs/SFT_RESULTS_REPORT.md` "LLM-judge" section). Ignore the
> "run to launch" below; kept for provenance. For what's next see `docs/HANDOFF_SFT_POST_JUDGE.md`.

**Created 2026-07-12.** For a fresh Claude session. Branch `feat/gemma4-12b-sft-poc-nvidia`.
Read alongside `docs/SFT_RESULTS_REPORT.md` (all results so far, incl. the consultant ablation +
oracle), `docs/SFT_VS_BASE_ANALYSIS_PLAN.md` (analysis menu — this is the **LLM-judge** item), and
`docs/HANDOFF_SFT_CONSULTANT_ABLATION.md` (the ablation lineage this follows). Live tracker: #130.
Use `uv run --no-sync` for Python.

## The one-sentence goal

The oracle arm showed the SFT writes better Socratic turns than base by **+22.7 ROUGE-1 / +19.1
BLEU-4** given correct state — but ROUGE only measures *overlap with the reference*. **Judge the two
oracle arms on an absolute pedagogical rubric (Claude Opus 4.8) to confirm the win holds on actual
quality, not just phrasing overlap.** Everything is built and verified; the run has **not** been
launched (waiting on a go-ahead).

## STATE: ready to run, not run

- Harness: `scripts/llm_judge_eval.py` — implemented, tested, committed (`af5c64b`, `95d49b1`),
  pushed. Full suite green (245 tests; 8 judge tests in `tests/test_llm_judge.py`).
- Verified live: a 5-turn subscription smoke on Opus 4.8 (sensible scores), and a `--dry-run` of the
  paired sampler on the real oracle arms (200/arm at `--per-stage 50`, no capping).
- **Nothing consumed the full subscription budget yet** — only the ~7-turn smoke.

## The run to launch (agreed parameters)

**`--per-stage 50 --stages bcde` → 200 turns/arm → 400 total Opus calls** (user chose this; it's the
sweet spot — firm per-stage estimates, overall verdict already settled, not worth going higher for an
effect this large). Sequential, ballpark ~1–1.5 h, consumes subscription (not API credits).

```
uv run --no-sync python scripts/llm_judge_eval.py \
  results/gemma4-12b-sft-oracle --compare-with results/gemma4-12b-base-oracle \
  --per-stage 50 --stages bcde --seed 42 \
  --out results/llm_judge_oracle_compare.json
```

Run it **detached** with a progress tail (the CLI prints one line per judged turn: `[arm] n/200
file#idx (stage): X/10`). Output: `results/llm_judge_oracle_compare.json` with per-arm
`overall_avg` / `axis_avgs` / `per_stage` / `cost_usd`, the `delta_a_minus_b` (SFT − base), and the
frozen `sample` list for reproducibility.

## How the harness works (so you can trust/debug it)

- **Backend = `claude-code` (default).** Judges via `claude -p --model claude-opus-4-8
  --output-format json --tools "" --permission-mode dontAsk --strict-mcp-config
  --disable-slash-commands`, prompt on stdin, parses `.result`. Verified against `claude` 2.1.207.
- **Subscription, NOT API key.** `ANTHROPIC_API_KEY` is *stripped from the child env* in
  `_claude_code_judge` because it takes precedence over the subscription and would silently bill the
  API. Confirm before running: `test -n "${ANTHROPIC_API_KEY:-}" && echo SET || echo unset` should
  say **unset**, and `claude` must be logged in to the subscription (a trivial `claude -p "say pong"
  --output-format json` should return `is_error:false`).
- **Sequential** — parallel `claude -p` would race one account. `--max-turns N` caps turns for a
  cheap smoke; `_parse_rubric` strips ``` fences and validates the 4 axes.
- **Rubric (0–10, absolute):** socratic_validity 0–3, advancement 0–3, age_appropriateness 0–2,
  question_form 0–2. See `JUDGE_SYSTEM_PROMPT`. The prompt shows the GT reference "as one valid move,
  do not penalize routing differences" **on purpose** — so the judge rewards *quality*, not
  ROUGE-like overlap (do not remove this, or the independence from ROUGE is lost).
- **Paired design:** `build_stratified_sample` keys on `(dialogue_file, turn_idx)`; both oracle arms
  replay identical GT student turns and hit `e34` at the same turn, so the sample is identical and
  comparable across arms. Stage `a` is excluded (trivial opener). Per-stage pools (headroom):
  b=761, c=1447, d=734, e=681 — so `--per-stage` up to ~680 before `e` caps.
- **Cost/tokens:** read from the CLI envelope's `total_cost_usd` (subscription-equivalent, not real
  billing); `input_tokens` is uncached-only, so the cache buckets are summed for a true token count.

## Gotchas

1. **Verify subscription auth first** (see above). If `ANTHROPIC_API_KEY` is set in the shell, either
   `unset` it or trust the in-code strip — but the child-env strip already handles it.
2. **Subscription usage limits.** 400 sequential Opus calls is modest but not free; if calls start
   failing with 429/401 the account hit a window/credit cap — pause and resume later (re-running
   re-judges from scratch; there's no turn-level checkpoint, so consider a smaller `--per-stage` or
   split across windows if limits bite).
3. **Determinism:** the sample is seeded (`--seed 42`), but Opus scoring itself isn't temperature-0
   here (subscription headless), so re-runs vary slightly. Report as a single run; multi-seed is a
   stretch, not required.
4. **`results/comparison.json`** is a stale regenerated artifact, dirty since several sessions ago —
   unrelated to this work; leave it or ignore it.

## Definition of done

`results/llm_judge_oracle_compare.json` exists with both arms scored on the 200/arm sample; the
SFT − base delta (overall + per-axis + per-stage) is written into **`docs/SFT_RESULTS_REPORT.md`**
(new "LLM-judge (absolute quality)" subsection, near the consultant-ablation section),
**`docs/EXPERIMENT_LOG.md`** (dated entry, newest-at-top), and **issue #130** (comment). State
plainly whether the judge **confirms** the ROUGE-based oracle result: SFT should beat base on
`socratic_validity` + `advancement` especially. If it *disagrees* (SFT matched reference phrasing but
isn't judged more pedagogically sound), that's a genuine caveat — surface it, don't bury it.

## Where things live

- **Harness:** `scripts/llm_judge_eval.py` (`--backend claude-code` default; `--compare-with` enables
  paired mode; `--per-stage`/`--stages`/`--seed`/`--out`/`--dry-run`/`--max-turns`). Tests:
  `tests/test_llm_judge.py`.
- **Inputs:** `results/gemma4-12b-{sft,base}-oracle/dialogues/*.json` (681 each, oracle arm — GT state
  fed in, so state_accuracy is 100% and the ROUGE/BLEU deltas are the signal).
- **Prior results:** `docs/SFT_RESULTS_REPORT.md` consultant-ablation section (Qwen / self-consult /
  oracle × base / SFT).
- **Commits this lineage:** `efa77a6`/`a99c2fd`/`7b85ff7` (oracle) → `af5c64b`/`95d49b1` (judge).

## After this: remaining analysis-plan picks

Multi-seed error bars; a strong-consultant (Claude-as-classifier) cell between the ~55–60% Qwen and
100% oracle state-quality points. Both optional — the SFT-vs-base story is already well-established.
