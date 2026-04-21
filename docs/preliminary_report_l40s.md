# Preliminary Evaluation Report — L40S Local Dual-GPU Run

**Date:** 2026-04-21
**Status:** Preliminary (n=25 sample)

---

## Setup

| Component | Value |
|---|---|
| Hardware | 2× NVIDIA L40S (48 GB each) |
| Teacher agent | SocratTeachLLM (GLM4-9B fine-tune, GPU 0 port 8001) |
| Consultant agent | Qwen3.5-9B (local vLLM, GPU 1 port 8002) |
| Config | `configs/l40s.env` |
| Max teaching rounds | 8 |
| Sample size | 25 / 681 test dialogues |
| Total turns evaluated | 148 |
| Run duration | 27 min (1591 s, ~64 s/dialogue) |

> **Note on consultant:** The original KELE paper (Peng et al., EMNLP 2025) used GPT-4o as the consultant agent for both dataset construction and evaluation. This run replaces GPT-4o with a fully local Qwen3.5-9B served via vLLM — no external API calls.

---

## Metrics

### Text Overlap (vs. ground-truth teacher responses)

| Metric | Score |
|---|---|
| ROUGE-1 | 43.49 |
| ROUGE-2 | 24.18 |
| ROUGE-L | 36.20 |
| BLEU-4 | 17.76 |

### State Classification Accuracy

| Stage | Accuracy |
|---|---|
| Overall | 18.92% |
| Stage a (student questioning) | 60.00% |
| Stage b (concept probing) | 25.00% |
| Stage c (inductive reasoning) | 10.64% |
| Stage d (rule construction) | 0.00% |
| Stage e (teacher summary) | 4.35% |

---

## Observations

- **Stage a accuracy is reasonable (60%)** — the earliest and simplest stage, where state transitions are most predictable.
- **Stages d and e are effectively failing (0% / 4.35%)** — these later stages require the consultant to issue precise structured outputs (state ID + evaluation + action). Qwen3.5-9B appears to drift from the expected format in later dialogue turns.
- **Overall state accuracy (18.92%) is well below what GPT-4o would produce** as the original consultant. The structured consultant prompt was designed around GPT-4o's instruction-following capability.
- **ROUGE scores are a secondary concern** — text overlap measures how closely the teacher's generated responses match ground truth, which is more a function of SocratTeachLLM than the consultant.
- **Throughput: ~64 s/dialogue** on dual L40S with sequential vLLM inference. Full 681-dialogue run would take ~12 hours.

---

## Next Steps

- Run full 681-dialogue evaluation (`make run-eval GPU=l40s`) for statistically meaningful results.
- Investigate consultant prompt tuning for Qwen3.5-9B to improve structured output adherence, particularly for stages d and e.
- Consider a larger local consultant model (e.g. Qwen3.5-32B) if VRAM allows.
- Compare against baseline (GPT-4o consultant) on the same 25 dialogues for a direct delta.

---

## Raw JSON

```json
{
  "n_turns": 148,
  "rouge1": 43.49,
  "rouge2": 24.18,
  "rougeL": 36.2,
  "bleu4": 17.76,
  "state_accuracy": {
    "overall": 18.92,
    "per_stage": {
      "a": 60.0,
      "b": 25.0,
      "c": 10.64,
      "d": 0.0,
      "e": 4.35
    },
    "total_turns": 148
  }
}
```
