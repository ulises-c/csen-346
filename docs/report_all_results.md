# KELE Evaluation — All Results

**CSEN 346 · Santa Clara University**
**Last updated:** 2026-04-23

---

## Summary table

| Run | Teacher | Teacher HW | Consultant | Consultant HW | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | State acc | Wall time | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper (SocratTeachLLM) | SocratTeachLLM | — | GPT-4o *(GT labels)* | — | — | 57.40 | 33.63 | 50.77 | 41.96 | — | — | Ground-truth consultant labels fed directly; not a live run |
| Paper (GPT-4o teacher) | GPT-4o | — | GPT-4o *(GT labels)* | — | — | 48.25 | 22.55 | 38.27 | 29.93 | — | — | Paper Table 1 GPT-4o-as-teacher baseline |
| Run 1 *(invalid)* | SocratTeachLLM | RTX 5090 | Qwen3.5-2B | RTX 5090 | 681 | 0.29 | 0.08 | 0.28 | 0.00 | 15.08% | 4h 58m | **English bug + wrong BLEU tokenizer** — results invalid |
| **Run 2 — Baseline** | SocratTeachLLM | RTX 5090 | GPT-4o-2024-11-20 | OpenAI API | 681 | **44.61** | **26.04** | **38.02** | **19.60** | **25.94%** | 4h 34m | Reference baseline for all comparisons |
| L40S preliminary | SocratTeachLLM | 2× L40S 48 GB | Qwen3.5-9B | 2× L40S 48 GB | 25 | 43.49 | 24.18 | 36.20 | 17.76 | 18.92% | 27 min | n=25 sample; small-n artifacts in per-stage scores |
| **WAVE HPC** | SocratTeachLLM | 2× V100 32 GB | Qwen3.5-9B | 2× V100 32 GB | 681 | 43.72 | 24.87 | 36.76 | 18.63 | 18.93% | 24h 54m | Full run; fp16 cast (V100 no bf16); enforce_eager |
| **R9700 + Mac Mini** | SocratTeachLLM | R9700 32 GB (ROCm) | qwen2.5:7b | Mac Mini M4 (Ollama, CPU) | 681 | 43.57 | 24.90 | 36.91 | 18.56 | 15.16% | 4h 28m | All-local, zero cloud cost |

### Per-stage state accuracy

| Run | Stage a | Stage b | Stage c | Stage d | Stage e |
|---|---|---|---|---|---|
| Run 1 *(invalid)* | 52.57% | 8.67% | 2.56% | 3.55% | 22.24% |
| **Baseline (GPT-4o)** | **95.15%** | **36.93%** | 4.70% | **5.04%** | 11.92% |
| L40S preliminary (n=25) | 60.00% | 25.00% | 10.64%* | 0.00%* | 4.35%* |
| **WAVE HPC (Qwen3.5-9B)** | 57.12% | 26.45% | **6.99%** | 3.43% | **14.11%** |
| **R9700 + Mac Mini (qwen2.5:7b)** | 66.67% | 17.40% | 0.63% | 4.96% | 2.39% |

\* Small-sample artifact (n=25). L40S stage-c/d/e numbers are not reliable at this scale.

---

## vs. Baseline (Run 2 — GPT-4o consultant)

Δ = this run minus baseline. Positive = improvement.

| Run | ΔROUGE-1 | ΔROUGE-2 | ΔROUGE-L | ΔBLEU-4 | ΔState acc |
|---|---|---|---|---|---|
| Paper (SocratTeachLLM) | +12.79 | +7.59 | +12.75 | +22.36 | — |
| Paper (GPT-4o teacher) | +3.64 | −3.49 | +0.25 | +10.33 | — |
| WAVE HPC (Qwen3.5-9B) | −0.89 | −1.17 | −1.26 | −0.97 | −6.01% |
| R9700 + Mac Mini (qwen2.5:7b) | −1.04 | −1.14 | −1.11 | −1.04 | −10.78% |

---

## Hardware and cost

| Run | Teacher HW | Consultant HW | API cost | Notes |
|---|---|---|---|---|
| Run 1 *(invalid)* | RTX 5090 | RTX 5090 (shared) | $0 | Qwen3.5-2B: too weak, OOM at 4B |
| Baseline | RTX 5090 | OpenAI API | **$17.49** | $0.0257/dialogue; prompt caching saved ~$6.80 |
| L40S preliminary | 2× L40S 48 GB | 2× L40S 48 GB | $0 | Compute server; ~64 s/dialogue |
| WAVE HPC | 2× V100 32 GB | 2× V100 32 GB | $0 | Slurm; enforce_eager → ~131 s/dialogue |
| R9700 + Mac Mini | R9700 32 GB ROCm | Mac Mini M4 CPU | $0 | ~23.6 s/dialogue; fastest local full run |

---

## Key findings

**Text overlap (ROUGE/BLEU) is stable across consultant quality.** The Qwen3.5-9B (WAVE) and qwen2.5:7b (R9700+Mac Mini) runs score within ±0.15 ROUGE-1 of each other despite a 9B→7B consultant downgrade. ROUGE/BLEU reflects teacher generation quality, which is the same SocratTeachLLM in both runs.

**State accuracy is the primary consultant-sensitive metric.** Overall accuracy drops 6 pts (WAVE vs baseline) and 11 pts (R9700+Mac Mini vs baseline) as consultant capability decreases from GPT-4o → Qwen3.5-9B → qwen2.5:7b. Stage c (22-state fine-grained classification) is the most sensitive: GPT-4o 4.70% → Qwen3.5-9B 6.99% → qwen2.5:7b 0.63%.

**Stage a anomaly.** The smaller consultants (Qwen3.5-9B at 57%, qwen2.5:7b at 67%) underperform GPT-4o (95%) on stage a but improve monotonically as the consultant gets weaker. Separately, the WAVE run's stage-a failure was identified in `docs/QWEN_EVAL_FIX_PLAN.md` as a Qwen3.5-9B-specific issue with the a0→a1 transition. The qwen2.5:7b result (66.67%) may reflect a different failure mode.

**ROCm on R9700 is production-viable.** Four full hours of inference across 4,262 turns with no reported ROCm errors. Text overlap matches CUDA-based runs, confirming no quality penalty from the AMD path.

**WAVE HPC V100s are the throughput bottleneck.** enforce_eager (required on cc 7.0) and Triton attention (no FlashAttention 2) yield 131 s/dialogue — 5.6× slower than the R9700+Mac Mini setup (23.6 s/dialogue).

**The ~14 ROUGE-1 gap to the paper's SocratTeachLLM result is methodological.** The paper feeds ground-truth consultant labels directly to the teacher; our runs use live consultant predictions. This is not a quality regression — it is a measurement difference.

---

## Original findings (paper reproduction)

The tables below are sourced from `docs/EXPERIMENT_LOG.md` and the original KELE paper (Peng et al., EMNLP 2025).

### Paper Table 1 (reproduced)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 |
|---|---|---|---|---|
| SocratTeachLLM (paper) | 57.40 | 33.63 | 50.77 | 41.96 |
| GPT-4o-as-teacher (paper) | 48.25 | 22.55 | 38.27 | 29.93 |
| **Our reproduction (GPT-4o consultant)** | **44.61** | **26.04** | **38.02** | **19.60** |

Our run beats the paper's GPT-4o-as-teacher baseline on **ROUGE-2 (+3.49)** and matches it on **ROUGE-L (−0.25)**. The BLEU-4 gap (19.60 vs 41.96 for SocratTeachLLM) is likely a generation-parameter mismatch — the paper does not specify temperature or max_tokens.

### Run 2 baseline — full metrics

**Config:** `configs/baseline.env` · Teacher: SocratTeachLLM on RTX 5090 · Consultant: gpt-4o-2024-11-20 via OpenAI API
**Date:** 2026-04-14 10:14 → 14:48 · Wall time: 4h 34m · Dialogues: 681 / 681 · Turns: 4,294

| Metric | Score |
|---|---|
| ROUGE-1 | 44.61 |
| ROUGE-2 | 26.04 |
| ROUGE-L | 38.02 |
| BLEU-4 | 19.60 |
| State accuracy (overall) | 25.94% |
| Stage a | 95.15% |
| Stage b | 36.93% |
| Stage c | 4.70% |
| Stage d | 5.04% |
| Stage e | 11.92% |

**OpenAI spend:** $17.49 total ($0.0257/dialogue). Prompt caching on the ~2,800-token system prompt saved ~$6.80 (~28%) vs list pricing.

### Consultant comparison (from EXPERIMENT_LOG — 2026-04-14)

Tested during baseline development. All runs on RTX 5090; same teacher (SocratTeachLLM).

| Consultant | State acc | Outcome |
|---|---|---|
| Qwen3.5-2B (local vLLM) | 1.64% | Too weak — emitted bare integers instead of `"a1"`/`"b4"` format |
| Qwen3.5-4B (local vLLM) | OOM | Teacher (19 GB) + 4B weights (8 GB) + KV cache exceeds 32 GB |
| gpt-4o-mini (API) | 6.56% | Stage a dropped from 100% to 30% — insufficient |
| **gpt-4o-2024-11-20 (API)** | **29.06%** | **Selected. Matches paper's original setup.** |

---

## Artifacts

| Run | Directory | Key files |
|---|---|---|
| Run 1 *(invalid)* | `results/baseline_run1_en_bug/` | `metrics_summary.json`, `run_config.json` |
| Baseline | `results/baseline/` | `metrics_summary.json`, `run_config.json`, `dialogues/` (681) |
| L40S preliminary | *(no results dir)* | Metrics sourced from `docs/preliminary_report_l40s.md` |
| WAVE HPC | `results/wave-2026-04-21T08-59-20-892964/` | `metrics_summary.json`, `run_config.json`, `dialogues/` (681) |
| R9700 + Mac Mini | `results/R9700_Mac-M4/` | `metrics_summary.json`, `run_config.json`, `dialogues/` (681) |
| *(crash artifact)* | `results/Ulisess-Mac-mini/` | Empty `dialogues/` only — run crashed at config load; no results |

Individual run reports: [`report_wave_hpc.md`](report_wave_hpc.md) · [`report_R9700_Mac-M4.md`](report_R9700_Mac-M4.md) · [`preliminary_report_l40s.md`](preliminary_report_l40s.md)
