# Preliminary Evaluation Report — Wave Cluster (Full Run)

**Date:** 2026-04-22
**Status:** Full run (n=681 test dialogues)

---

## Setup

| Component | Value |
|---|---|
| Hardware | 2× Tesla V100-PCIE-32GB |
| Teacher agent | SocratTeachLLM (ChatGLM fine-tune, GPU 0 port 8001) |
| Consultant agent | Qwen3.5-9B (local vLLM, GPU 1 port 8002) |
| Config | `configs/wave.env` |
| Max teaching rounds | 8 |
| Sample size | 681 / 681 test dialogues (full run) |
| Total turns evaluated | 4,280 |
| vLLM version | 0.19.1 |

### vLLM serve flags (both models)

| Flag | Value | Note |
|---|---|---|
| `dtype` | `float16` | V100 has no bfloat16 support; both models cast from bf16 |
| `max_model_len` | 8192 (consultant), 4096 (teacher) | |
| `enforce_eager` | `True` | CUDA graph capture disabled — required on V100; slower than L40S |
| `gpu_memory_utilization` | 0.85 | ~27–28 GB used per GPU at peak |

> Both models were trained/stored as bfloat16 and silently cast to float16 at load time.
> This is a V100 hardware constraint and may introduce minor numerical differences vs
> the L40S run (which supports bfloat16 natively).

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
  identical (18.92% vs 18.93%), validating the preliminary as a reliable proxy.
- **Stage d (0% → 3.43%) and stage e (4.35% → 14.11%) at full scale** — both were
  small-sample artifacts in the L40S preliminary. At n=681 they are in the expected
  range relative to the GPT-4o baseline.
- **Stage a (57%) remains the primary failure mode.** Qwen3.5-9B misses the
  `a0 → a1` student-question trigger ~43% of the time, consistent across both
  hardware runs and the full dataset. This cascades into all downstream stages.
- **Stage e (14.11%) exceeds GPT-4o (11.92%).** Once Qwen reaches e34, it does so
  at the right turn — the problem is not recognising stage transitions, but getting
  there via the wrong intermediate states.
- **V100 fp16 penalty is likely minor.** ROUGE/BLEU scores are within ±1 point of
  the L40S (bf16) run, suggesting the dtype cast does not meaningfully affect output.
- **`enforce_eager` slows throughput** — expected on V100 vs L40S; not a quality issue.

---

## Next Steps

See [`docs/QWEN_EVAL_FIX_PLAN.md`](QWEN_EVAL_FIX_PLAN.md) for the full action plan. Priority order:

1. **GT-consultant eval mode** — bypass live consultant with ground-truth outputs to
   isolate teacher quality and match the paper's Table 1 setup.
2. **Stage-a prompt hardening** — tighten the `a1` trigger in the consultant prompt
   for fill-in-the-blank and multiple-choice question formats (highest expected gain).
3. **JSON failure rate audit** — run `python -m src.project.debug --experiment wave`
   to measure how often Qwen3.5-9B produces malformed JSON under vLLM fp16.

---

## Raw JSON

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
