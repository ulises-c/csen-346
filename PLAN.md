# Project Plan — KELE Reproduction & Extension

**CSEN 346 · Santa Clara University**

## Status

| Item                            | Status         |
| ------------------------------- | -------------- |
| Paper selected & presented      | ✅ Done        |
| Repo structure set up           | ✅ Done        |
| KELE code obtained & translated | ✅ Done        |
| SocratDataset obtained          | ✅ Done        |
| Baseline running                | ⬜ Not started |
| Evaluation pipeline             | ⬜ Not started |
| Improvement implemented         | ⬜ Not started |
| Paper written                   | ⬜ Not started |

---

## Course Deadlines

| Date               | Deliverable                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| **Apr 14** | 1st documented GitHub commit + Paper: Intro & Related Work                          |
| **Apr 23** | 2nd documented GitHub commit + Paper: Dataset & Methodology                         |
| **May 5**  | 3rd documented GitHub commit + Paper: Evaluation & Results                          |
| **May 14** | 4th documented GitHub commit + Paper: Results, Intro, Conclusion, Limitations, Ethics |
| **May 26** | Demo & Present (~3 groups)                                                          |
| **May 28** | Demo & Present (~3 groups)                                                          |
| **Jun 2**  | Demo & Present (~3 groups)                                                          |
| **Jun 4**  | Final paper + final code + HuggingFace data + poster                                |

---

## Phase 1 — Environment & Baseline (Now → Apr 14)

**Goal:** Get the KELE system running end-to-end with SocratTeachLLM as the teacher agent.
**Hardware:** RTX 5090 32GB (all inference and training runs on this rig).
**Consultant model:** Qwen3.5-9B (replaces GPT-4o from original paper — fully local, zero paid APIs).
**Setup guide:** See [`RTX5090_SETUP.md`](RTX5090_SETUP.md) for full runbook — model downloads, vLLM serving, eval pipeline, metrics export, and troubleshooting.

- [x] Add `openai` to `pyproject.toml` via Poetry
- [ ] Serve [SocratTeachLLM](https://huggingface.co/yuanpan/SocratTeachLLM) locally (no free HF Inference API available)
- [ ] Configure API credentials for consultant (Qwen3.5-9B, local) and teacher (SocratTeachLLM, local)
- [ ] Run `KELE_original/consultant_teacher_socratic_teaching_system.py` on 5–10 manual test dialogues
- [ ] Verify the 5-stage SocRule flow (a → b → c → d → e) works correctly end-to-end
- [ ] Commit working baseline to `src/`

### Model Experiments (in order)

**Run 1 — SocratTeachLLM (baseline reproduction)**
- [ ] Serve SocratTeachLLM (9.4B, GLM4-9B fine-tune) via vLLM / MLX / Ollama
- [ ] Run full evaluation pipeline with this as the teacher agent
- [ ] Compare results against Table 1 in the paper

**Run 2 — Gemma-4-31B (extension experiment)**
- [ ] Serve [Gemma-4-31B](https://huggingface.co/google/gemma-4-31B) locally (Q4 quantized)
- [ ] Run the same evaluation pipeline with Gemma-4-31B as the teacher agent
- [ ] Compare against SocratTeachLLM — does a stronger general model beat a Socratic fine-tune?

### Hardware & Training Time Estimates

Two machines available: **M1 Max 64GB** (unified memory, MPS/MLX) and **RTX 5090 32GB** (CUDA).

#### Inference throughput (for evaluation runs)

| Model | M1 Max 64GB (MLX) | RTX 5090 32GB (vLLM) |
| --- | --- | --- |
| SocratTeachLLM 9.4B (BF16) | ~15–25 tok/s | ~80–120 tok/s |
| SocratTeachLLM 9.4B (Q4) | ~25–35 tok/s | ~100–150 tok/s |
| Gemma-4-31B (Q4, ~17GB) | ~5–10 tok/s | ~25–45 tok/s |
| Gemma-4-31B (Q8, ~33GB) | ~8–15 tok/s | ~15–30 tok/s (VRAM-tight) |

#### Full evaluation run estimates (681 test dialogues × ~6 turns × ~200 tok/response ≈ 816K tokens)

| Model | M1 Max 64GB | RTX 5090 32GB |
| --- | --- | --- |
| SocratTeachLLM 9.4B | ~6–10 hours | ~1.5–3 hours |
| Gemma-4-31B (FP4) | ~16–40 hours | ~4–8 hours |

#### Phase 4 — BERT classifier training (42K examples, 3–5 epochs)

| Model | M1 Max 64GB | RTX 5090 32GB |
| --- | --- | --- |
| distilbert-base | ~15–30 min | ~3–8 min |
| bert-base-uncased | ~25–45 min | ~5–12 min |

#### Decision: RTX 5090
All inference and training will run on the **RTX 5090 32GB** rig. The M1 Max cannot keep up with the Gemma-4-31B evaluation timeline (~40 hrs vs ~4–8 hrs). Running everything on one machine also avoids environment duplication.

**Files to create:**

- `src/project/kele.py` — working copy of the system extended from KELE_original
- `src/project/config.py` — API config loaded from environment variables (no hardcoded keys)

---

## Phase 2 — Evaluation Pipeline (Apr 14 → Apr 23)

**Goal:** Reproduce Table 1 from the paper. This phase produces the 1st and 2nd commits.

### Automated metrics (ROUGE-1/2/L, BLEU-4)

- [ ] Split `KELE_original/SocratDataset.json` into train (90%) and test (10%) — matching the paper
- [ ] Run the system on the test set and collect teacher responses
- [ ] Score with `rouge-score` and `sacrebleu` against ground-truth responses

### Single-turn metrics (PRR, NDAR, SPR, IAR)

- [ ] Implement GPT-4o-as-judge scorer using the evaluation criteria from Appendix B (paper)
- [ ] Run on a random sample of 100 single-turn dialogues from the test set
- [ ] Manually verify ~20 judgements for sanity

### Multi-turn metrics (Guidance, Logicality, Flexibility, Repetitiveness, Clarity)

- [ ] Implement GPT-4o multi-turn scoring using the prompt from Appendix B.5 (paper)
- [ ] Run on a sample of 100 multi-turn dialogues

### Ablation (confirms paper claims)

- [ ] Run teacher-only (no consultant) and compare — confirms consultant adds value
- [ ] Run GLM4-9B base vs. SocratTeachLLM — confirms fine-tuning helped

**Target:** Results within ~5% of Table 1 across all 13 metrics.

**Files to create:**

- `src/project/evaluate.py` — evaluation runner
- `src/project/metrics.py` — ROUGE, BLEU, and LLM-as-judge scorers
- `results/baseline_results.json` — raw evaluation output

---

## Phase 3 — Gap Analysis (Apr 23 → Apr 28)

**Goal:** Understand where our reproduction matches the paper and where it diverges.

- [ ] Document which metrics hit parity and which fall short
- [ ] Diagnose causes of gaps (consultant model choice, prompt differences, eval methodology)
- [ ] Confirm the paper's two stated limitations hold in our run:
  - PRR ~75% (~25% of turns have relevance issues)
  - Domain specificity (system trained only on elementary science)
- [ ] Write up findings as the baseline section of the paper

---

## Phase 4 — Improvement (Apr 28 → May 5)

**Goal:** Beat the baseline on at least one meaningful metric. This produces the 3rd commit.

### Recommended: Learned Consultant (replace LLM consultant with a classifier)

The consultant's task is a classification problem: given `(dialogue_history, student_input)`, predict `(stage, state_number)`. The SocratDataset already has these labels — it's a supervised learning dataset we already own.

- [ ] Extract `(history, student_input) → state` examples from the train split
- [ ] Fine-tune a lightweight classifier (`bert-base-uncased` or `distilbert`) on this task
- [ ] Swap the LLM consultant in `kele.py` with the trained classifier
- [ ] Re-run the full evaluation pipeline on the modified system

**Why this improvement matters:**

- Directly targets the paper's stated 25% rule non-compliance (PRR = 75.13)
- Replaces an expensive LLM call with a fast, deterministic classifier — lower latency and cost
- Tests whether structured state classification can outperform unstructured LLM reasoning

**Fallback (if classifier underperforms):** Cross-domain evaluation — build a small English test set (math or CS topics) and analyze how well KELE transfers. Propose what changes would improve cross-domain performance.

**Files to create:**

- `src/project/consultant_classifier.py` — classifier training and inference
- `src/project/kele_improved.py` — modified system using the classifier consultant
- `results/improved_results.json` — evaluation output for the improved system

---

## Phase 5 — Paper & Final Deliverables (May 5 → Jun 4)

**Goal:** Final paper, poster, demo, and presentation. This produces the 4th commit and final submission.

### Paper (4–6 pages, ACL template)

- [ ] Introduction & Related Work — due Apr 14
- [ ] Dataset & Methodology — due Apr 23
- [ ] Evaluation & Results — due May 5
- [ ] Conclusion, Limitations, Ethics — due May 14
- [ ] Final polish — due Jun 4
- [ ] Run through [Agentic Reviewer by Andrew Ng](https://www.agentic-reviewer.com/) before submitting

### Code submission checklist

- [ ] Docstrings on all functions and classes
- [ ] `.env.example` with required API key names documented
- [ ] README: model description, installation, usage, expected output, member contributions
- [ ] HuggingFace dataset card for any data artifacts

### Demo

- [ ] Record a short walkthrough of the system handling a full 5-stage dialogue
- [ ] Host on HuggingFace Spaces, YouTube, or Google Drive
- [ ] Reference the demo link in the paper

### Poster

- [ ] Follow the guideline: more images, minimal text
- [ ] Key panels: problem, KELE architecture, our improvement, results comparison table

---

## Improvement Options (Ranked by Feasibility)

| Option                              | Effort    | Expected Impact             | Notes                             |
| ----------------------------------- | --------- | --------------------------- | --------------------------------- |
| **Learned Consultant (classifier)** | Medium    | High — targets PRR directly | Uses existing dataset labels      |
| Cross-domain evaluation             | Low       | Medium — analysis only      | No new model, needs new test data |
| RL-based stage transitions          | High      | High                        | Complex, risky for timeline       |
| Mixture of Experts gating           | Very High | Speculative                 | Out of scope for 8 weeks          |

---

## Division of Work (suggested)

| Member   | Primary Area                                       |
| -------- | -------------------------------------------------- |
| Member 1 | Baseline setup, serving SocratTeachLLM, API config |
| Member 2 | Evaluation pipeline (ROUGE/BLEU + LLM-as-judge)    |
| Member 3 | Improvement (classifier consultant), paper writing |

All members contribute to paper writing and final presentation.
