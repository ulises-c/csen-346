# Experiment Log

Engineering decisions, what we've tried, and what's next. Each entry is dated and time-ordered (newest at top).

---

## 2026-04-14 — Baseline run #1 post-mortem + fixes

**Run:** `results/baseline/` (2026-04-13 23:32 → 2026-04-14 04:31, 681/681 dialogues, ~5h)

**Metrics (before fixes):**
| Metric | Value | Notes |
|---|---|---|
| ROUGE-1 / 2 / L | 0.29 / 0.08 / 0.28 | raw fmeasure — effectively 0 (English vs Chinese) |
| BLEU-4 | 0.0 | zero n-gram overlap, plus wrong tokenizer |
| State acc (overall) | 15.08% | stages b/c/d near-zero |

### Problems identified

1. **Language mismatch (critical).** SocratDataset is Chinese; ground-truth teacher turns are Chinese. The KELE system prompts in `resources/KELE/consultant_teacher_socratic_teaching_system.py` had been translated to English, so SocratTeachLLM replied in English. Zero overlap with references → BLEU/ROUGE collapse.
2. **sacrebleu tokenizer.** `compute_bleu` used the default `13a` (English) tokenizer. For Chinese we need `tokenize="zh"`.
3. **Consultant context overflow.** 111 consultant calls (2.8% of 3978 turns) hit the 4096-token limit on Qwen3.5-2B. Those turns fell back to "stay in current state", further depressing state accuracy.

### Fixes applied (2026-04-14)

- [x] `src/project/kele.py` — import from `resources/KELE/original_CN/consultant_teacher_socratic_teaching_system_CN.py` (Chinese system prompts).
- [x] `src/project/metrics.py` — `BLEU(effective_order=True, tokenize="zh")`.
- [x] `scripts/serve_consultant.sh` — `--max-model-len 4096 → 8192`. See Option A below.

### Next

Restart consultant vLLM with the new `--max-model-len`, then rerun baseline. Expect large jumps in BLEU-4 and ROUGE once teacher outputs Chinese; state accuracy should also rise as the consultant no longer falls back on truncated turns.

---

## Decision log — Consultant context window

**Context:** Qwen3.5-2B consultant is hitting 4096-token limits on long dialogues. System prompt alone is ~2,800 tokens (stage rules + state tables) before history/input.

### Options considered

| Option | Description | Effort | Risk | Status |
|---|---|---|---|---|
| **A** | Bump `--max-model-len` from 4096 → 8192 on consultant vLLM | 1-line change in `scripts/serve_consultant.sh` | Low — Qwen3.5-2B native context is 32k; GPU has headroom at 0.32 util | **In progress (2026-04-14)** |
| B | Swap consultant to Qwen2.5-7B-Instruct (32k native context, stronger reasoning) | ~1h — download + config + rerun | Medium — more VRAM, may compete with teacher on 5090 | Not tried |
| C | Truncate / summarize history in `get_full_formatted_history` | Medium | High — undermines stage-round tracking that the consultant depends on | Rejected |

**Plan:** Try A first. If state accuracy is still low after the language fix + A, move to B (2B consultant may also simply be too weak — the paper used GPT-4o).

---

## Budget — LLM calls for the full campaign

- Each turn: 2 LLM calls (consultant + teacher)
- 3,978 turns/run × 2 ≈ **~8,000 calls per full eval run**
- Planned runs: baseline (SocratTeachLLM) + Gemma-4 extension + BERT-consultant improvement = 3 minimum
- Reserve 1-2 reruns per experiment → **~30-40k calls total**
- All local on 5090 → ~5h wall-clock per run → **~20-30 GPU-hours** for the campaign
