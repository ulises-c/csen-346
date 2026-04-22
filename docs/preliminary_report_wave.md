# Preliminary Evaluation Report — Wave Cluster (Full Run)

**Date:** 2026-04-22
**Status:** Full run (n=681 test dialogues)
**Slurm job:** 892964 · node `gpu03` · submitted 2026-04-21 01:59:07 PDT

---

## Setup

| Component | Value |
|---|---|
| Hardware | 2× Tesla V100-PCIE-32GB (32 GiB each, compute capability 7.0) |
| CUDA driver | 580.126.20 |
| Teacher agent | SocratTeachLLM (GLM4-9B fine-tune, ChatGLMForConditionalGeneration) GPU 0 port 8001 |
| Consultant agent | Qwen3.5-9B (Qwen3_5ForConditionalGeneration) GPU 1 port 8002 |
| Config | `configs/wave.env` |
| vLLM version | 0.19.1 |
| Max teaching rounds | 8 |
| Dataset split | test (10%, seed=42) |
| Total dialogues | 681 / 681 completed |
| Total turns evaluated | 4,280 (avg 6.3 turns/dialogue) |
| Wall time | 24 h 53 m 34 s (02:01:55 → 02:55:29 next day) |
| Throughput | ~131.6 s/dialogue · ~4.1 dialogues/min |

### vLLM configuration

| Setting | SocratTeachLLM (teacher) | Qwen3.5-9B (consultant) |
|---|---|---|
| `dtype` | `float16` ¹ | `float16` ¹ |
| `max_model_len` | 4096 | 8192 |
| `enforce_eager` | `True` ² | `True` ² |
| `gpu_memory_utilization` | 0.85 | 0.85 |
| KV cache memory | 8.74 GiB (229,152 tokens) | — |
| Attention backend | TRITON_ATTN ³ | TRITON_ATTN ³ |
| Prefix caching | enabled | **disabled** |
| Chunked prefill | enabled | enabled |
| Sampling (teacher) | temperature=0.8, top_p=0.8 ⁴ | — |

¹ Both models are stored as bfloat16 and silently cast to float16 at load time — V100s
  lack native bfloat16 support. Scores are within ±1 point of the L40S (bf16) run,
  suggesting the cast has negligible quality impact.

² `enforce_eager` disables CUDA graph capture and `torch.compile`. Required on V100
  (compute capability 7.0 < 8.0). This is the primary driver of the slower throughput
  vs the L40S run (~131 s/dialogue here vs ~64 s/dialogue on L40S).

³ FlashAttention 2 is unavailable (requires cc ≥ 8.0). vLLM falls back to Triton
  attention automatically.

⁴ Sourced from SocratTeachLLM's `generation_config.json`, not from the serve command.

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

| Stage | Accuracy |
|---|---|
| Overall | 18.93% |
| Stage a (student questioning) | 57.12% |
| Stage b (concept probing) | 26.45% |
| Stage c (inductive reasoning) | 6.99% |
| Stage d (rule construction) | 3.43% |
| Stage e (teacher summary) | 14.11% |

---

## Comparison: Wave (full) vs. L40S (n=25) vs. GPT-4o baseline (full)

| Metric | Wave / Qwen3.5-9B (n=681) | L40S / Qwen3.5-9B (n=25) | GPT-4o baseline (n=681) |
|---|---|---|---|
| ROUGE-1 | 43.72 | 43.49 | 44.61 |
| ROUGE-2 | 24.87 | 24.18 | 26.04 |
| ROUGE-L | 36.76 | 36.20 | 38.02 |
| BLEU-4 | 18.63 | 17.76 | 19.60 |
| Overall state acc. | 18.93% | 18.92% | 25.94% |
| Stage a | 57.12% | 60.00% | **95.15%** |
| Stage b | 26.45% | 25.00% | 36.93% |
| Stage c | 6.99% | 10.64% | 4.70% |
| Stage d | 3.43% | 0.00%* | 5.04% |
| Stage e | **14.11%** | 4.35%* | 11.92% |

\* Small-sample artifact (n=25).

---

## Observations

- **The n=25 L40S preliminary was accurate.** Overall state accuracy is virtually
  identical (18.92% vs 18.93%), validating the L40S preliminary as a reliable proxy
  for full-run results.
- **Stage d (0% → 3.43%) and stage e (4.35% → 14.11%) were small-sample artifacts.**
  At full scale both stages are in the expected range relative to the GPT-4o baseline.
- **Stage a (57.12%) is the primary failure mode.** Qwen3.5-9B misses the
  `a0 → a1` student-question trigger ~43% of the time, consistent across both
  hardware targets and the entire test set. Because stage a seeds all downstream
  state decisions, this error cascades into every subsequent stage.
- **Stage e (14.11%) exceeds the GPT-4o baseline (11.92%).** Once the system reaches
  `e34` via whatever path, it does so at the correct turn — the problem is the
  intermediate route, not the final transition.
- **Stage c is worse at full scale (6.99%) than the n=25 sample (10.64%).** The
  preliminary figure was optimistic noise from the small sample; 6.99% is more
  representative of how both models perform on the 22-state c-stage classification.
- **fp16 cast has negligible quality impact.** All ROUGE/BLEU scores are within
  ±1 point of the L40S (bf16) run.
- **Prefix caching is disabled on the consultant.** Qwen3.5-9B is a Mamba-hybrid
  architecture and vLLM disabled prefix caching automatically. This means every
  consultant call processes the full growing dialogue history from scratch, contributing
  to the slow ~131 s/dialogue throughput.
- **Slow tokenizer on SocratTeachLLM.** vLLM logged a warning; this adds latency
  on the teacher side and is worth resolving in future runs.

---

## Next Steps

See [`docs/QWEN_EVAL_FIX_PLAN.md`](QWEN_EVAL_FIX_PLAN.md) for the full action plan. Priority order:

1. **GT-consultant eval mode** — bypass live consultant with ground-truth outputs to
   isolate teacher quality and match the paper's Table 1 setup.
2. **Stage-a prompt hardening** — tighten the `a1` trigger in the consultant prompt
   for fill-in-the-blank and multiple-choice question formats (highest expected gain).
3. **JSON failure rate audit** — run `python -m src.project.debug --experiment wave`
   to measure how often Qwen3.5-9B produces malformed JSON under vLLM fp16.
4. **Slow tokenizer** — investigate replacing SocratTeachLLM's slow tokenizer with
   a fast equivalent to reduce per-turn latency in future runs.

---

## Raw JSON

`results/wave-2026-04-21T08-59-20-892964/metrics_summary.json`
```json
{
  "n_turns": 4280,
  "rouge1": 43.72,
  "rouge2": 24.87,
  "rougeL": 36.76,
  "bleu4": 18.63,
  "state_accuracy": {
    "overall": 18.93,
    "per_stage": {
      "a": 57.12,
      "b": 26.45,
      "c": 6.99,
      "d": 3.43,
      "e": 14.11
    },
    "total_turns": 4280
  }
}
```

`results/wave-2026-04-21T08-59-20-892964/run_config.json`
```json
{
  "experiment": "wave",
  "teacher_model": "SocratTeachLLM",
  "teacher_base_url": "http://127.0.0.1:8001/v1",
  "consultant_model": "Qwen3.5-9B",
  "consultant_base_url": "http://127.0.0.1:8002/v1",
  "max_teaching_rounds": 8,
  "total_dialogues": 681,
  "completed": 681,
  "total_elapsed_seconds": 89614.07,
  "started_at": "2026-04-21 02:01:55",
  "finished_at": "2026-04-22 02:55:29"
}
```
