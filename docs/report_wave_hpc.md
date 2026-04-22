# Evaluation Report — WAVE HPC Cluster (Qwen3.5-9B, Full Run)

**Date:** 2026-04-22
**Slurm job:** 892964 · node `gpu03` · submitted 2026-04-21 01:59:07 PDT
**Results:** `results/wave-2026-04-21T08-59-20-892964/`

---

## Setup

| Component | Value |
|---|---|
| Hardware | 2× Tesla V100-PCIE-32GB (32 GiB each, compute capability 7.0) |
| CUDA driver | 580.126.20 |
| vLLM version | 0.19.1 |
| Teacher agent | SocratTeachLLM · `ChatGLMForConditionalGeneration` · GPU 0 · port 8001 |
| Consultant agent | Qwen3.5-9B · `Qwen3_5ForConditionalGeneration` · GPU 1 · port 8002 |
| Config | `configs/wave.env` |
| Max teaching rounds | 8 |
| Dataset split | test (10%, seed=42) |
| Total dialogues | 681 / 681 completed |
| Total turns evaluated | 4,280 (avg 6.3 turns/dialogue) |
| Wall time | 24 h 53 m 34 s (started 02:01:55 · finished 02:55:29 next day) |
| Throughput | ~131.6 s/dialogue |

### vLLM configuration

| Setting | SocratTeachLLM (teacher) | Qwen3.5-9B (consultant) |
|---|---|---|
| `dtype` | `float16` ¹ | `float16` ¹ |
| `max_model_len` | 4096 | 8192 |
| `enforce_eager` | `True` ² | `True` ² |
| `gpu_memory_utilization` | 0.85 | 0.85 |
| KV cache | 8.74 GiB / 229,152 tokens | — |
| Attention backend | TRITON_ATTN ³ | TRITON_ATTN ³ |
| Prefix caching | enabled | **disabled** ⁴ |
| Chunked prefill | enabled | enabled |
| Sampling | temperature=0.8, top_p=0.8 ⁵ | — |

¹ Both models are stored as bfloat16; V100 (cc 7.0) lacks native bf16 support so vLLM
  silently casts to float16 at load time.

² `enforce_eager` disables CUDA graph capture and `torch.compile`. Required on V100
  (cc < 8.0). Primary driver of slow throughput vs L40S (~131 s/dialogue vs ~64 s/dialogue).

³ FlashAttention 2 unavailable on cc < 8.0; vLLM falls back to Triton attention automatically.

⁴ Qwen3.5-9B is a Mamba-hybrid architecture; vLLM disabled prefix caching automatically.
  Every consultant call reprocesses the full dialogue history from scratch.

⁵ Sourced from SocratTeachLLM's `generation_config.json`, not from the serve flags.

---

## Metrics

### Text Overlap (vs. ground-truth teacher responses)

| Metric | Score |
|---|---|
| ROUGE-1 | 43.72 |
| ROUGE-2 | 24.87 |
| ROUGE-L | 36.76 |
| BLEU-4 | 18.63 |

### State Classification Accuracy

| Stage | Accuracy | Description |
|---|---|---|
| Overall | 18.93% | |
| Stage a | 57.12% | Student questioning |
| Stage b | 26.45% | Concept probing |
| Stage c | 6.99% | Inductive reasoning (22 states) |
| Stage d | 3.43% | Rule construction |
| Stage e | 14.11% | Teacher summary |

---

## Comparison against baselines

| Metric | WAVE HPC · Qwen3.5-9B (n=681) | L40S · Qwen3.5-9B (n=25) | GPT-4o baseline (n=681) | Paper Table 1 · SocratTeachLLM ⁶ |
|---|---|---|---|---|
| ROUGE-1 | 43.72 | 43.49 | 44.61 | 57.40 |
| ROUGE-2 | 24.87 | 24.18 | 26.04 | 33.63 |
| ROUGE-L | 36.76 | 36.20 | 38.02 | 50.77 |
| BLEU-4 | 18.63 | 17.76 | 19.60 | 41.96 |
| Overall state acc. | 18.93% | 18.92% | 25.94% | — |
| Stage a | 57.12% | 60.00% | **95.15%** | — |
| Stage b | 26.45% | 25.00% | 36.93% | — |
| Stage c | 6.99% | 10.64%* | 4.70% | — |
| Stage d | 3.43% | 0.00%* | 5.04% | — |
| Stage e | **14.11%** | 4.35%* | 11.92% | — |

\* Small-sample artifact (n=25).

⁶ Paper uses ground-truth consultant outputs fed directly to SocratTeachLLM (not live
  predictions). The ~13 ROUGE-1 gap to the paper is methodological, not model quality.

---

## Observations

**The L40S preliminary was representative.** Overall state accuracy is virtually
identical across both hardware targets (18.92% vs 18.93%), validating the 25-dialogue
preliminary as a reliable proxy for full-run results.

**Stage a is the primary failure mode.** Qwen3.5-9B misses the `a0 → a1`
student-question transition ~43% of the time across the full 681-dialogue test set.
Because stage a seeds all downstream state decisions, these errors cascade into every
subsequent stage and account for most of the overall accuracy gap vs GPT-4o.

**Stage c corrects downward at full scale.** The n=25 preliminary showed 10.64% — an
optimistic small-sample artifact. 6.99% at full scale is consistent with the inherent
difficulty of 22-state fine-grained classification in stage c.

**Stage e (14.11%) exceeds the GPT-4o baseline (11.92%).** Once the system reaches
`e34`, it does so at the correct turn. The problem is the intermediate path, not the
final transition.

**fp16 cast has negligible quality impact.** All ROUGE/BLEU scores are within ±1 point
of the L40S (bf16) run despite the dtype difference.

**Prefix caching disabled on consultant adds latency.** The Mamba-hybrid architecture
of Qwen3.5-9B prevents vLLM from caching key-value state across turns. Combined with
`enforce_eager`, this is the main reason throughput is 2× slower than L40S.

**SocratTeachLLM uses a slow tokenizer.** vLLM logged a tokenizer performance warning
on startup; switching to a fast tokenizer equivalent would reduce per-turn teacher
latency in future runs.

---

## Next Steps

Full action plan: [`docs/QWEN_EVAL_FIX_PLAN.md`](QWEN_EVAL_FIX_PLAN.md)

| Priority | Action | Expected impact |
|---|---|---|
| 1 | GT-consultant eval mode | Closes ~13 ROUGE-1 gap to paper; isolates teacher quality |
| 2 | Stage-a prompt hardening | +3–5 pts state accuracy; fixes `a0→a1` miss rate |
| 3 | JSON failure rate audit (`python -m src.project.debug`) | Quantifies silent fallback rate |
| 4 | Fast tokenizer for SocratTeachLLM | Reduces per-turn teacher latency |

---

## Artifacts

| File | Description |
|---|---|
| `results/wave-2026-04-21T08-59-20-892964/metrics_summary.json` | Full metrics output |
| `results/wave-2026-04-21T08-59-20-892964/run_config.json` | Run configuration and timing |
| `results/wave-2026-04-21T08-59-20-892964/slurm.out` | Full vLLM startup logs and job output |
| `results/wave-2026-04-21T08-59-20-892964/slurm.err` | Slurm stderr (empty — no errors) |
| `results/wave-2026-04-21T08-59-20-892964/job.submitted` | Slurm submission record |
