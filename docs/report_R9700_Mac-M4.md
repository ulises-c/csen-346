# Evaluation Report — R9700 + Mac Mini M4 (qwen2.5:7b Consultant)

**Date:** 2026-04-23
**Results:** `results/R9700_Mac-M4/`

---

## Setup

| Component | Value |
|---|---|
| Hardware — teacher | AMD Radeon AI PRO R9700 · 32 GB VRAM · RDNA 4 (gfx1201) · ROCm |
| Hardware — consultant | Mac Mini M4 · CPU inference via Ollama |
| Teacher agent | SocratTeachLLM · `ChatGLMForConditionalGeneration` · localhost:8001 |
| Consultant agent | qwen2.5:7b · Ollama · `Ulisess-Mac-mini.local:11434` |
| Config | `configs/R9700_Mac-M4.env` |
| Max teaching rounds | 8 |
| Dataset split | test (10%, seed=42) |
| Total dialogues | 681 / 681 completed |
| Total turns evaluated | 4,262 (avg 6.26 turns/dialogue) |
| Wall time | 4 h 27 m 40 s (12:43:15 → 17:10:55 PDT) |
| Throughput | ~23.6 s/dialogue |

### Key configuration differences vs. prior runs

| Setting | This run | WAVE HPC (Qwen3.5-9B) | GPT-4o baseline |
|---|---|---|---|
| Consultant | qwen2.5:7b · Ollama (CPU, Mac M4) | Qwen3.5-9B · vLLM (V100 GPU) | GPT-4o API |
| Teacher inference | ROCm · R9700 · vLLM | CUDA · V100 · vLLM | CUDA · V100 · vLLM |
| Consultant context | 16,384 tokens | 8,192 tokens | API default |
| Thinking mode | enabled (not suppressed) | enabled | N/A |

---

## Metrics

### Text Overlap (vs. ground-truth teacher responses)

| Metric | Score |
|---|---|
| ROUGE-1 | 43.57 |
| ROUGE-2 | 24.90 |
| ROUGE-L | 36.91 |
| BLEU-4 | 18.56 |

### State Classification Accuracy

| Stage | Accuracy | Description |
|---|---|---|
| Overall | 15.16% | |
| Stage a | 66.67% | Student questioning |
| Stage b | 17.40% | Concept probing |
| Stage c | 0.63% | Inductive reasoning (22 states) |
| Stage d | 4.96% | Rule construction |
| Stage e | 2.39% | Teacher summary |

---

## Comparison against all baselines

| Metric | **R9700 · qwen2.5:7b** | WAVE · Qwen3.5-9B (n=681) | GPT-4o baseline (n=681) | Paper · SocratTeachLLM ¹ |
|---|---|---|---|---|
| ROUGE-1 | 43.57 | 43.72 | 44.61 | 57.40 |
| ROUGE-2 | 24.90 | 24.87 | 26.04 | 33.63 |
| ROUGE-L | 36.91 | 36.76 | 38.02 | 50.77 |
| BLEU-4 | 18.56 | 18.63 | 19.60 | 41.96 |
| Overall state acc. | **15.16%** | 18.93% | 25.94% | — |
| Stage a | **66.67%** | 57.12% | 95.15% | — |
| Stage b | 17.40% | 26.45% | 36.93% | — |
| Stage c | **0.63%** | 6.99% | 4.70% | — |
| Stage d | 4.96% | 3.43% | 5.04% | — |
| Stage e | **2.39%** | 14.11% | 11.92% | — |

¹ Paper uses ground-truth consultant outputs fed directly to SocratTeachLLM (not live predictions). The ~14 ROUGE-1 gap to the paper is methodological, not model quality.

---

## Observations

**Text overlap scores are essentially equivalent to WAVE HPC despite a 9B→7B consultant downgrade.** ROUGE-1 (43.57 vs 43.72), ROUGE-2 (24.90 vs 24.87), ROUGE-L (36.91 vs 36.76), and BLEU-4 (18.56 vs 18.63) are all within ±0.15 of the full Qwen3.5-9B run. Teacher output quality is not sensitive to consultant model size in the 7B–9B range — the teacher's own generation drives text overlap, and SocratTeachLLM is the same in both runs.

**State accuracy drops significantly: 15.16% vs 18.93%.** The ~3.8 point overall drop is entirely explained by the weaker consultant. qwen2.5:7b at 7B parameters cannot reliably perform the structured 30-state classification that the pipeline requires, particularly in the complex mid-to-late stages.

**Stage a accuracy improves: 66.67% vs 57.12% (WAVE).** This is the one area where qwen2.5:7b outperforms Qwen3.5-9B. Stage a involves the simplest state transitions (a0→a1, a1→a2), and the smaller model may be less likely to overthink and misclassify them. This is consistent with the Qwen3.5-9B-specific finding in `docs/QWEN_EVAL_FIX_PLAN.md` that the 9B model misses `a0→a1` ~43% of the time.

**Stage c collapses to near-zero: 0.63% vs 6.99% (WAVE).** Stage c requires 22-state fine-grained classification — the hardest classification task in the pipeline. qwen2.5:7b cannot reliably handle this. This is the strongest argument against using a sub-10B consultant.

**Stage e also collapses: 2.39% vs 14.11% (WAVE).** Stage e requires the consultant to correctly identify when to conclude the dialogue. The 7B model fails at this almost entirely.

**Throughput is 5.6× faster than WAVE HPC: 23.6 s/dialogue vs 131.6 s/dialogue.** Two contributing factors: (1) qwen2.5:7b on the Mac M4 responds significantly faster than Qwen3.5-9B on V100 + enforce_eager; (2) the R9700 (RDNA 4, ROCm) is a modern GPU without the `enforce_eager` penalty that constrained the V100. Total wall time was 4h28m vs 24h54m for a full-scale run.

**ROCm on R9700 is production-viable for SocratTeachLLM inference.** The teacher ran stably for 4,262 turns without reported ROCm errors. Text overlap scores matching CUDA runs confirms no quality degradation from the AMD path.

---

## Summary

This run establishes a **local-only baseline**: no cloud GPUs, no paid API, teacher on the R9700 via ROCm, consultant on the Mac Mini M4 via Ollama. The result shows that teacher output quality (ROUGE/BLEU) is preserved at this configuration, but state accuracy suffers significantly from the weaker consultant — confirming that consultant model quality is the primary lever for state classification accuracy, while the teacher model is the primary lever for text overlap.

The 66.67% stage-a accuracy (highest of any run on this metric) is a notable datapoint: a smaller, less capable consultant may paradoxically be better at simple binary transitions, which aligns with the Qwen3.5-9B over-complexity failure mode documented in `QWEN_EVAL_FIX_PLAN.md`.

---

## Next steps

| Priority | Action | Expected impact |
|---|---|---|
| 1 | Upgrade consultant to qwen3.5:9b on Mac Mini (Ollama pull) | Recover ~3.8 pts state accuracy; stage c/e most affected |
| 2 | GT-consultant eval mode | Closes ~14 ROUGE-1 gap to paper; isolates teacher quality |
| 3 | Hierarchical BERT classifier (Improvement #2) | Replace Ollama consultant with local 125M classifier; state acc +20–35 pts |
| 4 | Audit stage-c failures | 0.63% suggests systematic format failure, not just classification error |

---

## Artifacts

| File | Description |
|---|---|
| `results/R9700_Mac-M4/metrics_summary.json` | Full metrics output |
| `results/R9700_Mac-M4/run_config.json` | Run configuration and timing |
| `results/R9700_Mac-M4/dialogues/` | 681 saved dialogue files |
| `results/R9700_Mac-M4/progress.log` | Per-dialogue progress log |
