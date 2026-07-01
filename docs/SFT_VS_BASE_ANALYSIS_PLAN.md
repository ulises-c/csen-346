# SFT vs Base — Further Analysis & Ablation Plan

**Created 2026-06-24.** Roadmap for digging past the headline result into *what* the Socratic
QLoRA SFT changed and *how solid* the uplift is. Companion to `docs/SFT_HANDOFF.md` (pipeline +
provenance) and `docs/EXPERIMENT_LOG.md` (results log). Live eval tracker: GitHub issue **#130**.

## What we already know (baseline for everything below)

- **Headline (ZH held-out test, n=681):** SFT **state_acc 59.93** vs base **49.62** (MTP-off) /
  **50.30** (MTP-on) → **+9.6–10.3 pp**, ~13–14σ at the assumed σ≈0.7 pp. Per-stage gains +10–18 pp
  on b/c/d/e; a was already 100. ROUGE/BLEU also ~2–4× (style learned). Clean run, 0 errors, ~3 h.
- **In progress (2×2 generalization):** base & SFT on EN held-out test (~680) + ZH/EN synthetic OOD
  (75 each, never trained). Results → `results/gemma4-12b-{base,sft}-{en,synth-zh,synth-en}`.
- **The adapter** is `checkpoint-4250` (~0.88 epoch, recovered from a NaN divergence). HF:
  `ulises-c/SocratesLM-12B` (merged BF16), `-GGUF` (Q8_0), `-QLoRA` (adapter + ckpts 3200..4250).
- **Eval invariants (held fixed, base vs SFT):** Qwen3.5-0.8B LoRA state classifier as consultant
  (CPU), 8 teaching rounds, workers=4, MTP off, `-np 4`/q4_0 KV, **stochastic** server-default
  sampling (no temperature/seed set in the teacher call, `tournament_utilizations.py:440`).
  Only the teacher model and the dataset change.

## Open questions this plan answers

1. **What did the SFT internalize** vs just "use the consultant better"? (ablation #1)
2. **Is it a better *teacher***, not just a better state-emitter? (LLM-judge)
3. **How real is +9.6** — error bars, sampling-noise-free, not overfit? (multi-seed, greedy, train-gap, earlier ckpt)
4. **Did the narrow SFT cost general ability** or depend on the quant? (capability, quant)

---

## Tier 0 — Free: analyze dialogues already on disk (no GPU)

Operate on the saved `results/<run>/dialogues/*.json`. Each turn record has `state`,
`ground_truth_state`, `teacher_response`, `ground_truth_teacher`.

- **T0.1 — Termination / length / Socratic-style metrics** ⭐ (explains the OOD rambling we saw)
  Per turn: response length, **truncation rate** (hit `max_tokens=2048` with no EOS), and
  **questions-per-turn** (reuse `tournament_utilizations.validate` — single clean question, no
  preamble). Hypothesis: SFT is shorter, single-question, terminates cleanly; base rambles on OOD.
- **T0.2 — State confusion matrix (base vs SFT)** — which stage transitions improved (esp. the
  +17.9 pp on stage e). Surfaces whether base collapses to a few stages.
- **T0.3 — Train-vs-test gap** — eval SFT on the **train** split (`--split train`); a large
  train≫test gap flags overfitting at 0.88 epoch.
- **T0.4 — Slice ZH-test uplift by grade/subject** — `SocratDataset` carries `grade`/`chapter`/
  `mission`; check whether the gain is uniform or topic-concentrated.

## Tier 1 — Highest-value new runs

- **T1.1 — No-consultant ablation** ⭐⭐ (the key missing control)
  Re-run base vs SFT **without** `--bert-consultant` so the teacher gets no externally-supplied
  state. The SFT was trained in the inference format that emits state/action itself; this tests
  whether the +9.6 is the SFT's own internalized state-tracking or just better use of the
  classifier (which currently aids *both* arms equally).
  ```
  # bare teacher, no classifier — run base then SFT, same machinery otherwise
  KELE_BERT_DEVICE=cpu KELE_PARALLEL_WORKERS=4 uv run --no-sync python -m src.project.kele \
    --experiment gemma4-12b-{local,sft-local} evaluate --output results/gemma4-12b-{base,sft}-noconsult
  ```
  (Serve the matching model first; no `--bert-consultant` flag.)
- **T1.2 — LLM-as-judge on Socratic quality** ⭐ — pairwise *blind* judge (e.g. Claude) over a
  sample of base vs SFT teacher turns: "which guides *without giving away the answer*?" Captures
  pedagogy that state-acc/ROUGE miss. Build on the existing Claude-consultant configs +
  `tournament_utilizations` style critic. New small harness.

## Tier 2 — Harden the headline number

- **T2.1 — Multi-seed error bars** — σ≈0.7 is from one pair. Re-run ZH-test 2–3× (vary decode;
  `--sample-seed` reshuffles the subsample, or rely on stochastic decode) to get a real stdev on
  +9.6. ~3 h each.
- **T2.2 — Greedy (temp=0) run** — removes sampling noise for a clean point estimate. **Needs a
  small change**: thread a `temperature`/`seed` through the teacher call (currently unset →
  server default). Worth adding as a general eval knob.
- **T2.3 — Earlier-checkpoint comparison** — eval `checkpoint-3200` (loss plateaued ~step 3000) vs
  shipped `checkpoint-4250`. If saturated at 3200, more epochs won't help; if still climbing, a
  clean full run is justified. Reuse `merge_lora_gemma4_sft.py` + `convert_gemma4_12b_sft_to_gguf.sh`
  on the earlier adapter (both ckpts on HF `ulises-c/SocratesLM-12B-QLoRA`).

## Tier 3 — Different axes

- **T3.1 — Capability preservation** — small general-QA probe (non-Socratic) on base vs SFT to
  rule out catastrophic forgetting from the narrow QLoRA.
- **T3.2 — Quant sensitivity** — spot-check merged **BF16** (transformers) vs the **Q8_0 GGUF** on
  ~50 dialogues to confirm the uplift isn't a quant artifact (validates the "Q8 delta ≈ noise"
  assumption). Optionally Q4/Q5 to see how far the gain survives compression.

---

## Suggested order (value / cost)

1. **T0.1–T0.4** now — free, run on existing dialogues while the 2×2 finishes.
2. **T1.1 no-consultant** — the one control the current design lacks.
3. **T1.2 LLM-judge** + **T2.1 multi-seed** — quality evidence + error bars for the writeup.
4. **T2.3 earlier-checkpoint** — decides whether a cleaner full-epoch run is worth ~30 h.
5. **T3.x** — nice-to-have insurance.

If only three: **T1.1 (no-consultant)**, **T1.2 (LLM-judge)**, **T0.1 (style/termination)** — then
**T2.1 (multi-seed)** to firm up the number.

## Machinery notes

- Dataset/model selection already wired: `--hf-repo`/`--split` on `evaluate`; monitor honors
  `EVAL_HF_REPO`/`EVAL_SPLIT`/`EVAL_OUT_SUFFIX` (crash-resilient path for the unstable box).
- Each new eval set needs **both** base and SFT runs; base is ~6× slower than SFT on OOD synthetic
  (rambles to max_tokens), so budget base runs generously.
- `temperature`/`seed` are **not** currently passthrough on the teacher call — T2.2 (and any
  determinism work) needs that small addition first.
- Cosmetic TODO: the monitor's #130 progress rows hardcode `/681` as the denominator; patch
  `dataset_total` to read the actual split size for non-default datasets.
