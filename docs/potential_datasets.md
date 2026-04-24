# Potential Datasets for KELE Extension

**CSEN 346 · Santa Clara University**

Candidate datasets for expanding training data, evaluation, or cross-domain transfer experiments. Organized by relevance to the project.

---

## Tier 1 — Direct fit (Chinese math Socratic/tutoring dialogue)

### SocratDataset *(already using)*
- **Source:** Peng et al. EMNLP 2025 (KELE paper) — `yuanpan/SocratDataset` on HF
- **Size:** 6,803 dialogues, 42,000+ annotated turns
- **Language:** Chinese (Mandarin)
- **Why:** Ground-truth source for all current experiments. 30-state cognitive labels + 34 SocRule strategies. No substitute exists for this specific task.
- **Limitation:** ~6,800 dialogues is small; rare stages (c/d) underrepresented.

### Ape210K
- **Source:** Zhao et al. 2020 — [GitHub](https://github.com/Chenny0808/ape210k)
- **Size:** 210,488 Chinese elementary math word problems with solutions
- **Language:** Chinese
- **Why:** Large-scale source for Chinese math question content; useful for counterfactual dialogue augmentation (Improvement #7) and seeding student-simulator personas.
- **Limitation:** Not dialogue; single-turn Q&A only. Would need dialogue wrapping.

### CMATH
- **Source:** Wei et al. 2023 — `weitianwen/cmath` on HF
- **Size:** 1,700 Chinese middle-school math problems (MCQ + free response)
- **Language:** Chinese
- **Why:** Middle-school scope directly matches SocratDataset domain (algebra, geometry, probability). Could be used to generate new Socratic dialogues via GPT-4o.
- **Limitation:** Small; no dialogue annotations.

---

## Tier 2 — English math tutoring dialogue (cross-lingual transfer)

### MathDial *(already cited)*
- **Source:** Macina et al. EMNLP 2023 — `eth-nlped/mathdial` on HF
- **Size:** 3,000+ tutoring dialogues grounded in math reasoning problems
- **Language:** English
- **Why:** Rich pedagogical annotations; grounded reasoning; structural similarities to SocratDataset. Could inform classifier architecture design.
- **Limitation:** English only; domain shift from Chinese math curriculum.

### MathInstruct
- **Source:** Yue et al. 2023 — `TIGER-Lab/MathInstruct` on HF
- **Size:** 260,000 math instruction-following examples compiled from 13 datasets
- **Language:** English
- **Why:** Covers diverse math reasoning styles useful for augmentation. Includes chain-of-thought data that aligns with DeepSeek-R1-Distill's reasoning format.
- **Limitation:** Not dialogue or tutoring; instruction-response pairs only.

### GSM8K
- **Source:** Cobbe et al. 2021 — `openai/gsm8k` on HF
- **Size:** 8,500 grade-school math word problems with step-by-step solutions
- **Language:** English
- **Why:** Standard benchmark for math reasoning; widely used for evaluating consultant/teacher model upgrades.
- **Limitation:** No tutoring structure; single-turn solutions.

---

## Tier 3 — General Chinese instruction / tutoring

### EduChat Training Data
- **Source:** Dan et al. 2023 — arXiv:2308.02773
- **Size:** ~200,000 educational dialogue examples (mix of Chinese and English)
- **Language:** Chinese + English
- **Why:** Directly comparable system (EduChat) fine-tuned on educational dialogue. Training data may overlap thematically with SocratDataset.
- **Limitation:** Not publicly released in full; must request from authors.

### BELLE-Math
- **Source:** BELLE Team 2023 — `BelleGroup/train_2M_CN` on HF
- **Size:** 2M Chinese instruction-following pairs; math subset ~100k
- **Language:** Chinese
- **Why:** High-quality Chinese instruction data useful for pre-fine-tuning base models before SocratDataset LoRA.
- **Limitation:** Not tutoring/dialogue; general instruction following.

### MentorQA
- **Source:** AIM-SCU — `AIM-SCU/MentorQA` on HF
- **Size:** QA pairs from mentoring/advising dialogues
- **Language:** English (primarily)
- **Why:** Mentioned in course guidelines as a reference dataset; mentorship domain adjacent to Socratic teaching.
- **Limitation:** Mentorship/advising context, not math; English.

---

## Tier 4 — Evaluation benchmarks (for measuring base model improvement)

### C-Eval
- **Source:** Huang et al. 2023 — `ceval/ceval-exam` on HF
- **Size:** 13,948 multiple-choice questions across 52 subjects
- **Language:** Chinese
- **Why:** Standard benchmark for comparing Chinese LLM capability (used in IMPROVEMENT_PLAN.md model comparisons). Useful for verifying that fine-tuning doesn't degrade base model performance.

### CMMLU
- **Source:** Li et al. 2023 — `haonan-li/cmmlu` on HF
- **Size:** 11,528 questions across 67 subjects, Chinese cultural context
- **Language:** Chinese
- **Why:** More China-specific than C-Eval; strong signal for Chinese curriculum knowledge relevant to SocratDataset topics.

### AGIEval
- **Source:** Zhong et al. 2023 — `baber/agieval` on HF
- **Size:** 8,816 questions from Chinese/English college entrance exams (gaokao, SAT, etc.)
- **Language:** Chinese + English
- **Why:** Gaokao math section directly overlaps with SocratDataset middle/high school content.

---

## Synthetic augmentation sources

| Method | Source | Expected size | Notes |
|---|---|---|---|
| Counterfactual SocratDataset | GPT-4o generation from train split | 3× = ~18,000 dialogues | See IMPROVEMENT_PLAN.md #7 |
| Ape210K → dialogue wrapping | GPT-4o wraps Q&A into Socratic turns | Up to 50,000 dialogues | Quality filter needed |
| Student-simulator self-play | Trained student-sim + teacher rollouts | Unbounded | Requires student-sim (IMPROVEMENT_PLAN.md #9) |

---

## Summary recommendation

**For the current scope (June 4 deadline):**
- Stick with SocratDataset for all fine-tuning and evaluation — it has the cognitive-state labels no other dataset provides.
- Use CMATH or Ape210K if GPT-4o counterfactual augmentation (Improvement #7) is pursued.
- Use C-Eval/CMMLU to validate that the new base model (DeepSeek-R1-Distill-Qwen-14B or Qwen3-14B) hasn't been degraded by LoRA fine-tuning.

**Cross-lingual stretch goal:** MathDial offers the closest structural parallel in English; could be used for zero-shot transfer experiments if the classifier architecture needs English validation.
