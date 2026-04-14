# Project Plan — KELE Reproduction & Extension
**CSEN 346 · Santa Clara University**

## Status

| Item | Status |
|---|---|
| Paper selected & presented | ✅ Done |
| Repo structure set up | ✅ Done |
| KELE code obtained & translated | ✅ Done |
| SocratDataset obtained | ✅ Done |
| Baseline running | ⬜ Not started |
| Evaluation pipeline | ⬜ Not started |
| Improvement implemented | ⬜ Not started |
| Paper written | ⬜ Not started |

---

## Course Deadlines

| Date | Deliverable |
|---|---|
| **May 5–7** | 1st documented GitHub commit + Paper: Intro & Related Work |
| **May 12–14** | 2nd documented GitHub commit + Paper: Dataset & Methodology |
| **May 19–21** | 3rd documented GitHub commit + Paper: Evaluation & Results |
| **May 26–28** | 4th documented GitHub commit + Paper: Results, Conclusion, Limitations, Ethics |
| **May 26 – Jun 4** | Demo & presentation (W9–10, ~3 groups/session) |
| **Jun 4** | Final paper + final code + HuggingFace data + poster |

---

## Phase 1 — Environment & Baseline (Now → May 2)

**Goal:** Get the KELE system running end-to-end with SocratTeachLLM as the teacher agent.

- [ ] Add `openai` to `pyproject.toml` via Poetry
- [ ] Serve [SocratTeachLLM](https://huggingface.co/yuanpan/SocratTeachLLM) via HuggingFace Inference API or vLLM on SCU compute
- [ ] Configure API credentials for consultant (GPT-4o or Qwen2.5-14B) and teacher (SocratTeachLLM)
- [ ] Run `KELE_original/consultant_teacher_socratic_teaching_system.py` on 5–10 manual test dialogues
- [ ] Verify the 5-stage SocRule flow (a → b → c → d → e) works correctly end-to-end
- [ ] Commit working baseline to `src/`

**Files to create:**
- `src/project/kele.py` — working copy of the system extended from KELE_original
- `src/project/config.py` — API config loaded from environment variables (no hardcoded keys)

---

## Phase 2 — Evaluation Pipeline (May 2 → May 12)

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

## Phase 3 — Gap Analysis (May 12 → May 16)

**Goal:** Understand where our reproduction matches the paper and where it diverges.

- [ ] Document which metrics hit parity and which fall short
- [ ] Diagnose causes of gaps (consultant model choice, prompt differences, eval methodology)
- [ ] Confirm the paper's two stated limitations hold in our run:
  - PRR ~75% (~25% of turns have relevance issues)
  - Domain specificity (system trained only on elementary science)
- [ ] Write up findings as the baseline section of the paper

---

## Phase 4 — Improvement (May 16 → May 23)

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

## Phase 5 — Paper & Final Deliverables (May 23 → Jun 4)

**Goal:** Final paper, poster, demo, and presentation. This produces the 4th commit and final submission.

### Paper (4–6 pages, ACL template)
- [ ] Introduction & Related Work — due May 5
- [ ] Dataset & Methodology — due May 12
- [ ] Evaluation & Results — due May 19
- [ ] Conclusion, Limitations, Ethics — due May 26
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

| Option | Effort | Expected Impact | Notes |
|---|---|---|---|
| **Learned Consultant (classifier)** | Medium | High — targets PRR directly | Uses existing dataset labels |
| Cross-domain evaluation | Low | Medium — analysis only | No new model, needs new test data |
| RL-based stage transitions | High | High | Complex, risky for timeline |
| Mixture of Experts gating | Very High | Speculative | Out of scope for 8 weeks |

---

## Division of Work (suggested)

| Member | Primary Area |
|---|---|
| Member 1 | Baseline setup, serving SocratTeachLLM, API config |
| Member 2 | Evaluation pipeline (ROUGE/BLEU + LLM-as-judge) |
| Member 3 | Improvement (classifier consultant), paper writing |

All members contribute to paper writing and final presentation.
