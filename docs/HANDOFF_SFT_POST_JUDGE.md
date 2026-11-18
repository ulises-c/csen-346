# Handoff — SFT-vs-Base PoC: post-judge (hardening + remaining picks)

**Created 2026-07-22.** For a fresh Claude session. Branch `feat/gemma4-12b-sft-poc-nvidia`.
Read alongside `docs/SFT_RESULTS_REPORT.md` (the single source of truth for all results),
`docs/SFT_VS_BASE_ANALYSIS_PLAN.md` (the analysis menu), and `docs/SFT_HANDOFF.md` (pipeline +
box hazards). Live tracker: issue #130. Use `uv run --no-sync` for Python.

## Where the PoC stands (the story is COMPLETE and consistent)

The core scientific question — *did the Socratic QLoRA SFT of Gemma 4 12B actually learn something,
and what?* — is answered on every axis we set out to test. Nothing below is required to state the
result; it's all hardening / insurance.

- **Headline (with the shared Qwen classifier):** SFT beats base on state accuracy everywhere —
  ZH +10.3, EN +7.7 in-distribution; +3.5–3.9 OOD synthetic; 0 regressions. (~0.88-epoch adapter.)
- **Consultant ablation (T1.1), all three modes done** — Qwen classifier / self-consult / oracle ×
  base / SFT, ZH test, 681/681 each:
  - Teacher-turn *quality* is intrinsic and **largest with the state confound removed** (oracle:
    +22.7 ROUGE-1, +19.1 BLEU-4). The SFT climbs monotonically with state quality (ROUGE-1
    44.2→48.1→51.8); base is flat (28.0→28.6→29.0) — the SFT *learned to condition on state*.
  - State-*tracking* was entirely the classifier: self-consult SFT (26.8) is **worse** than base
    (34.5) — the SFT was trained to *consume* state, never emit it (`dataset.py:608–647`).
- **LLM-judge (T1.2), done** — Opus 4.8, absolute 0–10 rubric, 200 turns/arm paired on the oracle
  arms. **Confirms direction, corrects magnitude:** SFT +0.38/10 overall (wins/ties every axis &
  stage), but that's ~5% relative vs ROUGE's ~78% — most of the ROUGE gap was *reference-phrasing
  overlap*, not pedagogy. Real judge-visible edge = **question_form (+0.20)** and the **middle
  stages c (+0.64) / d (+0.76)**; `advancement` is a dead tie; endpoints tie. `results/llm_judge_oracle_compare.json`.

**One-line takeaway to preserve:** the SFT genuinely writes better-*formed* Socratic questions
(esp. mid-dialogue) and phrases turns corpus-style, but given correct state it is not dramatically
more *pedagogically* effective than base, and it does not self-track state. In deployment it wants
an external state source and rewards a better one with better turns.

## What's actually left (ranked; all optional)

### 1. T0.1 — Termination / length / style metrics ⭐ FREE, do this first
The report asserts "base rambles to the 2048-token cap, SFT terminates cleanly" (`SFT_RESULTS_REPORT.md:65–68`)
but it's **still unquantified** — the only open claim in the report. Now doubly worth closing because
the judge cost data corroborates it (base cost more to judge: $15.02 vs $13.86 = more tokens) and the
judge's **question_form** win begs a mechanical counterpart. No GPU — operate on the saved
`results/<run>/dialogues/*.json` (each turn has `teacher_response`, `state`, `ground_truth_*`).
Compute per turn, base vs SFT, across the runs already on disk:
- response length (chars/tokens), **truncation rate** (hit `max_tokens=2048` with no EOS),
- **questions-per-turn** — reuse `tournament_utilizations.validate` (single clean question, no preamble).
Hypothesis: SFT shorter, single-question, terminates; base rambles, esp. OOD. Write a small analysis
script (no new eval run), land results in the report's `## What this means` §4 (replace "to be
quantified") + a log entry. ~1–2 h of pure analysis.

### 2. T2.1 — Multi-seed error bars
Every number (ROUGE +9.6/+10.3, judge +0.38) is a **single-run point estimate**; σ≈0.7 pp is from one
pair. Re-run ZH-test base+SFT 2–3× (stochastic decode reshuffles; or `--sample-seed`) for a real
stdev on the headline. ~3 h/run on the box (SFT fast, base ~17 h — budget accordingly). Turns the
"~11–15σ" claim from assumed into measured.

### 3. T2.3 — Earlier-checkpoint comparison (the one *strategic* decision)
Eval `checkpoint-3200` (loss plateaued ~step 3000) vs shipped `checkpoint-4250` (both on HF
`ulises-c/SocratesLM-12B-QLoRA`). **Decides whether a clean full-epoch run (~30 h) is worth it:**
saturated at 3200 → more epochs won't help; still climbing → a full run is justified. Reuse
`merge_lora_gemma4_sft.py` + `convert_gemma4_12b_sft_to_gguf.sh` on the earlier adapter, then eval.

### 4. Lower value / insurance
- **T2.2 greedy (temp=0)** — clean point estimate, but **needs a small code change first**:
  `temperature`/`seed` are NOT passthrough on the teacher call (server default). Add that knob (it
  also enables cleaner multi-seed). Worth doing as general infra even independent of T2.2.
- **Strong-consultant cell** — Claude-as-classifier between the ~55–60% Qwen and 100% oracle
  state-quality points; bounds how much a better external state source lifts each teacher.
- **T3.1 capability preservation** (general-QA probe, catastrophic-forgetting check),
  **T3.2 quant sensitivity** (BF16 vs Q8_0 on ~50 dialogues — confirm the uplift isn't a quant
  artifact).

If you do only one: **T0.1** (free, closes the last open claim, complements the judge's question_form
finding). If prepping a writeup: **T0.1 + T2.1**. If deciding on more training: **T2.3**.

## Box + tooling hazards (unchanged, carry forward)

- **Unstable RTX 4000 Ada.** Stable at **85 W**; power step-down is **inert** (no passwordless sudo,
  so the monitor's "step down per crash" logs but can't actually change power — it just auto-resumes).
  **One model at a time on the 20 GB card.** GPU eval runs on this box are long and crash-prone — use
  the crash-resilient monitor, never bare `kele evaluate` for a full run.
- **Monitor:** `scripts/monitor_eval_gemma4_12b.sh {base|sft}` owns serve+eval, auto-resumes per
  dialogue, logs to #130. Env toggles: `EVAL_HF_REPO`/`EVAL_SPLIT`/`EVAL_OUT_SUFFIX` (other datasets),
  `NO_CONSULTANT=1` (self-consult), `ORACLE_CONSULTANT=1` (oracle). Chain wrappers:
  `scripts/{noconsult,oracle}_chain_gemma4_12b.sh` (`make {noconsult,oracle}-chain-gemma4-12b`) run
  SFT→base back-to-back detached. Cosmetic bug: #130 progress rows hardcode `/681` denominator —
  patch `dataset_total` for non-default split sizes.
- **LLM judge** (`scripts/llm_judge_eval.py`): backend `claude-code` (subscription, NOT API) —
  `ANTHROPIC_API_KEY` is stripped from the child env; verify `claude` is logged in first. Sequential
  (parallel would race one account), **no turn-level checkpoint** (a failed run re-judges from
  scratch — split `--per-stage` if usage limits bite). Rubric shows the GT reference "as one valid
  move, do not penalize routing" *on purpose* — keep it, or independence from ROUGE is lost.
- **`results/comparison.json`** has been dirty for many sessions — a stale regenerated artifact,
  unrelated to any current work. Leave it; don't stage it.
- **Commits go to two remotes** (`ulises-c/csen-346` + `SCU-CSEN346/KELE`); a plain `git push`
  pushes both. Pre-commit runs ruff+pyright+codespell+shellcheck+full pytest and **blocks** on
  failure. **Re-`git add` after fixing a pre-commit failure** — the hooks check the working tree but
  git commits the *staged* blob; a fix made without re-staging commits the stale version (this bit us
  in `efa77a6`, fixed in `46f84d6`).

## Where things live

- **Report (read first):** `docs/SFT_RESULTS_REPORT.md` — headline, per-stage, consultant ablation
  (3 modes), LLM-judge subsection, caveats, artifacts.
- **Results on disk:** `results/gemma4-12b-{base,sft}{,-en,-synth-zh,-synth-en,-noconsult,-oracle}/`
  + `-base-mtp`; `results/llm_judge_oracle_compare.json`.
- **Models (HF, private):** `ulises-c/SocratesLM-12B-QLoRA` (adapter, ckpts 3200–4250),
  `-12B` (merged BF16), `-12B-GGUF` (Q8_0, served).
- **Key code:** `src/project/kele.py` (`create_system` + eval loop; consultant modes incl.
  `--oracle-consultant`), `src/project/socratic_teaching_system.py` (base + `SocraticTeachingSystemOracle`),
  `scripts/llm_judge_eval.py`, the monitor + chain wrappers.
- **Superseded handoffs:** `HANDOFF_LLM_JUDGE.md` (its run is DONE — ignore its "run to launch"),
  `HANDOFF_SFT_CONSULTANT_ABLATION.md` (T1.1, done).

## Definition of done (for whichever pick you take)

Result lands in `docs/SFT_RESULTS_REPORT.md` (the right existing section), a dated newest-at-top
entry in `docs/EXPERIMENT_LOG.md`, and an issue #130 comment — the established pattern. For T0.1
specifically: replace the "to be quantified" hedge in `## What this means` §4 with the measured
numbers.
