# Dataset Expansion Plan

**CSEN 346 · Santa Clara University**

Goal: grow beyond the 6,803-record SocratDataset by (a) generating new Socratic dialogues from English question banks and (b) augmenting coverage of underrepresented states and subjects.

---

## 1. The KELE dataset construction pipeline (from Appendix B.3 & B.4)

The authors built SocratDataset using a fully LLM-driven two-prompt pipeline — no human annotation required beyond a final review pass. The pipeline is:

```
Raw MCQ corpus
    ↓  B.3 — filter unsuitable questions + optimize metadata
    ↓  B.4 — generate full SocRule-compliant Socratic dialogue
    →  SocratDataset record
```

### B.3: Problem Filtering and Solution Information Optimization

Applied to every raw question before dialogue generation. Does two things:

**Filter** — rejects questions that cannot support multi-turn Socratic sub-questioning:
- Questions that only describe physical appearance (visual memorization, no reasoning)
- Fact/historical questions (no room for exploration)
- Diagram-dependent questions (options refer to figures not included in text)

In the paper, 6,000 of 12,000 CSQ questions passed (≈50% pass rate). Expect similar rates on other MCQ banks.

**Optimize** — rewrites or fills in three metadata fields used by B.4:
- `hint`: a concise guiding clue, should not give away the answer
- `knowledgePoint`: precise academic description of the relevant concept(s)
- `problemSolvingThought`: full analysis of the question and each answer option

### B.4: SocratDataset Construction

Generates the full multi-turn dialogue given the filtered, optimized question. Key constraints enforced by the prompt:
- Dialogue strictly follows SocRule stages a→b→c→d→e (no skipping, no regression)
- Only stage d may ask for the correct answer
- Student simulates multiple incorrect answers throughout
- Teacher asks exactly one question per turn, never gives away the answer directly
- Output is a JSON array of turns, each with: `student`, `evaluation`, `state`, `action`, `teacher`

Both prompts are fully self-contained templates — swapping the question corpus or LLM requires no structural changes.

---

## 2. Candidate English question banks

| Dataset | HF ID | Size | Format | Notes |
|---|---|---|---|---|
| ARC-Easy | `allenai/ai2_arc` | ~2.3K | MCQ, 4 options | Elementary/middle school science; clean text |
| OpenBookQA | `allenai/openbookqa` | ~5K | MCQ + `fact1` | `fact1` maps directly to `hint`; strong baseline |
| SciQ | `allenai/sciq` | ~14K | MCQ + `support` | School-level science; `support` usable as `knowledgePoint` seed |
| MMLU (science) | `lighteval/mmlu` | varies | MCQ | Subsets: `elementary_science`, `high_school_biology`, etc. |

After B.3 filtering (expect ~50% pass), these yield roughly:

| Source | Est. records after filter |
|---|---|
| ARC-Easy | ~1,100 |
| OpenBookQA | ~2,500 |
| SciQ | ~7,000 |
| MMLU science subsets | ~1,000–3,000 |

---

## 3. Expansion strategies

### 3a. Direct pipeline replication (highest priority)

Run the B.3→B.4 pipeline on ARC-Easy or OpenBookQA to generate a SocratDataset-compatible English corpus natively — no translation step required.

```
ARC-Easy / OpenBookQA
    ↓  B.3 filter + enrich (Claude API or GPT-4o batch)
    ↓  B.4 dialogue gen (same LLM)
    →  SocratDataset-EN-generated
```

Publish alongside the translated SocratDataset-EN as a second split or separate HF dataset.

### 3b. Multiple dialogue paths per question

The current SocratDataset has exactly one dialogue per question. Running B.4 N times per question produces N diverse student misconception paths, each taking different branches through the b→c→d state machine. This:
- Multiplies dataset size without new questions
- Increases coverage of underrepresented states (Figure 3 in the paper shows heavy skew toward c9, c12, c15, d33, e34)
- Gives the model exposure to more of the 34 strategies during training

Practically: run B.4 3× per question with `temperature > 0` to get variation.

### 3c. Subject expansion

The paper explicitly lists this as a limitation. SocRule and the B.4 prompt are domain-agnostic. Candidate subjects:

| Subject | Source | Why it fits |
|---|---|---|
| Elementary math | `gsm8k` (filtered to grade 1–5 word problems) | Multi-step reasoning, clear sub-questions |
| Middle school science | MMLU `high_school_biology`, `earth_science` | Extends grade coverage |
| Social studies / civics | `civics_qa` | Different reasoning type (cause/effect) |

Math requires a prompt note that the teacher should not use formulas until stage d; otherwise B.4 applies unchanged.

### 3d. Augment with existing English tutoring datasets

Datasets with compatible pedagogical structure that could be merged or used for multi-task training:

| Dataset | Focus | Compatibility |
|---|---|---|
| MathDial | Math tutoring dialogues | Similar multi-turn structure; no SocRule states |
| EduChat | General education dialogues | Broad coverage; no structured stages |
| SocraticLM training data | Socratic Q&A | Closest; no SocRule annotation |

These are complementary, not direct replacements — they lack the SocRule state/action labels. Useful for curriculum learning or as a pre-fine-tune step before SocratDataset fine-tuning.

---

## 4. Quality control (replacing human review)

The original authors used graduate student review to validate dialogues. For at-scale generation, use the B.5 multi-turn evaluation rubric as an automated quality gate.

Filter: keep only dialogues scoring ≥ 3/5 on **Guidance** and **Logicality** (the two structural dimensions most affected by weak models). Evaluate with GPT-4o or a capable judge model.

Expected pass rate with a strong generator (GPT-4o or Claude Sonnet): ~80–90%.

---

## 5. Dataset statistics and structure

### Dialogue turn distribution

| Turns | Count | % of dataset |
|---|---|---|
| 5 | 1,740 | 25.6% |
| 6 | 1,740 | 25.6% |
| 7 | 1,471 | 21.6% |
| 8 | 591 | 8.7% |
| 9 | 256 | 3.8% |
| 10 | 58 | 0.9% |
| 11 | 25 | 0.4% |
| 12 | 6 | 0.1% |

Over 70% of dialogues are 5–6 turns. Longer dialogues (8+) occur when a student repeatedly misunderstands and the model cycles through multiple c-stage strategies.

### State/strategy distribution

The strategy distribution is heavily skewed (Figure 3). Every dialogue starts at a1 and ends at e34 — so both appear 6,803 times. The c-stage (Inductive Reasoning, states c8–c29) dominates because it is the core teaching stage and can repeat many times within one dialogue. The b-stage and d-stage are brief (the paper recommends 1–2 turns each), so their states appear less frequently.

**Practical implication:** A model fine-tuned on this data will have seen far fewer b-stage and d-stage examples than c-stage. When generating new data (Section 3b), deliberately over-sample questions where students struggle in early stages (concept probing) to balance the distribution.

### Per-turn data fields

Each dialogue turn in the JSON contains:

| Field | Type | Description |
|---|---|---|
| `student` | string | Student utterance (in turn 0: the original question + options) |
| `evaluation` | string | Consultant's assessment: current stage + state + justification |
| `state` | string | State code (a1, b2–b7, c8–c29, d30–d33, e34) |
| `action` | string | Teaching action the teacher should take (maps to one of 34 strategies) |
| `teacher` | string | Teacher's actual Socratic response |

The `evaluation` and `action` fields are the consultant's output — they are auxiliary training signals, not just metadata. See Section 6 for how they are used during training.

---

## 6. How to use SocratDataset for fine-tuning

### Training objective

The paper formulates teacher fine-tuning as maximizing the conditional probability of each teacher response given:
- All prior dialogue turns (history)
- The consultant's evaluation of the current student turn
- The consultant's recommended action

```
P(teacher_response | dialogue_history, evaluation, action)
```

This is **not** a simple seq2seq from student → teacher. The evaluation and action fields are critical conditioning signals.

### Input construction at turn n

```
input = {
    history:    [(s_1, t_1), (s_2, t_2), ..., (s_{n-1}, t_{n-1})],
    student:    s_n,          # current student turn
    evaluation: e_n,          # consultant's stage/state assessment
    action:     a_n           # consultant's recommended strategy
}
output = t_n                  # teacher response to generate
```

At inference time you need the consultant agent running in parallel — it produces `e_n` and `a_n` from the dialogue history before the teacher agent generates `t_n`. Without the consultant outputs as conditioning, the fine-tuned teacher model will underperform.

### Fine-tuning recipe (from the paper)

| Setting | Value |
|---|---|
| Base model | GLM4-9B |
| Method | LoRA |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Batch size | 16 |
| Train/test split | 90% / 10% (≈6,123 / 680 dialogues) |
| Hardware | 2× NVIDIA A800 GPU |

For English replication with the translated dataset, the same settings should transfer. If using a different base model (e.g., Qwen2.5-7B instead of GLM4-9B), the LoRA rank and target modules may need adjustment.

### Dataset splits

The test set used in the paper is **680 multi-turn dialogues**, decomposed into **4,245 single-turn examples** for automated evaluation. Using the same 90/10 split on SocratDataset-EN will give a directly comparable English benchmark.

---

## 7. Evaluation framework

### Single-turn metrics (binary)

Evaluated automatically on each teacher turn; ground truth from human raters:

| Metric | Abbreviation | Question answered |
|---|---|---|
| Problem Relevance Rate | PRR | Is the teacher's question directly related to the problem-solving process and the student's current input? |
| No Direct Answer Rate | NDAR | Does the teacher avoid giving away the answer, offering only minimal guidance? |
| Summary Pass Rate | SPR | In the final (e) stage, does the teacher provide a correct and complete summary? |
| Instruction Adherence Rate | IAR | Does the teacher strictly follow the consultant's recommended action? |

SPR and IAR are specific to structured Socratic teaching — they require SocRule annotations and cannot be computed on unstructured tutoring datasets.

**Human validation kappa scores (inter-rater agreement):**
PRR: 0.65 · NDAR: 0.71 · SPR: 0.75 · IAR: 0.75

### Multi-turn metrics (1–5 scale, GPT-4o judge)

Evaluated on the complete dialogue using the B.5 rubric prompt:

| Metric | What it measures |
|---|---|
| Guidance | Do questions guide students toward independent thinking, without giving away answers? |
| Logicality | Is the question sequence logically structured and cognitively progressive? |
| Flexibility | Does the teacher adapt to student responses rather than following a fixed script? |
| Repetitiveness | Are questions varied, avoiding mechanical repetition? |
| Clarity | Are questions clear and accessible at the student's level? |

**Human validation ICC (Inter-Class Correlation):**
Guidance: 0.72 · Logicality: 0.70 · Flexibility: 0.68 · Repetitiveness: 0.75 · Clarity: 0.83

### Performance benchmarks (Table 1)

Use these as baselines when evaluating a new English fine-tuned model:

| Model | ROUGE-1 | ROUGE-2 | BLEU-4 | PRR | NDAR | SPR | IAR | Guidance | Logicality | Flexibility |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | 38.25 | 22.35 | 29.93 | 72.13 | 81.19 | 85 | 87.74 | 4.35 | 4.50 | 4.33 |
| Qwen2.5-7B | 40.95 | 15.27 | 24.96 | 59.02 | 80.52 | 60 | 76.45 | 3.87 | 3.96 | 3.87 |
| Qwen2.5-14B | 43.79 | 17.06 | 26.63 | 65.21 | 78.57 | 74 | 80.81 | 3.99 | 4.15 | 4.03 |
| Qwen2.5-32B | 46.22 | 19.90 | 28.85 | 65.57 | 83.13 | 81 | 84.68 | 4.12 | 4.44 | 4.21 |
| SocraticLM-7B | 18.63 | 5.56 | 10.93 | 26.83 | 30.26 | 36 | 27.05 | 2.62 | 2.88 | 2.78 |
| EduChat-13B | 34.75 | 9.91 | 21.11 | 47.62 | 90.73 | 51 | 69.02 | 2.93 | 3.42 | 3.18 |
| **SocratTeachLLM** | **57.4** | **33.63** | **41.96** | **75.13** | **94.71** | **87** | **89.03** | **4.66** | **4.53** | **4.45** |

Key observations:
- SocratTeachLLM (GLM4-9B fine-tuned) beats GPT-4o on every metric despite being ~40× smaller
- PRR is the hardest metric — even SocratTeachLLM only reaches 75.13, showing rule-following is a remaining challenge
- EduChat achieves high NDAR (90.73) because it is overly cautious and avoids direct answers, but scores low on guidance — high NDAR alone is not sufficient
- ROUGE/BLEU scores are poor proxies for teaching quality; models like GPT-4o score lower than the fine-tuned model on text overlap despite having better general capability

---

## 8. Known limitations and gaps to address

| Limitation | Paper's note | Implication for expansion |
|---|---|---|
| Science-only | Explicitly flagged; future work needed | Priority for subject expansion (Section 3c) |
| PRR gap (75.13) | Rule-following still imperfect even after fine-tuning | Stricter B.4 prompt constraints + B.5 filtering may help |
| State distribution skew | c-stage dominates, b/d underrepresented | Multi-path generation should oversample b/d transitions |
| Single language (Chinese) | Dataset not tested in English | SocratDataset-EN translation is the first step |
| Single base model tested | Only GLM4-9B fine-tuned | Qwen2.5-7B/14B are natural candidates for English |
| No open-domain questions | Only elementary science | Subject expansion required for general Socratic tutoring |

---

## 9. Recommended sequencing

| Step | Action | Status |
|---|---|---|
| 1 | Translate SocratDataset (Chinese → English) | In progress — full run running |
| 2 | Publish `ulises-c/SocratDataset-EN` to HuggingFace | Blocked on step 1 |
| 3 | Run B.3 on ARC-Easy + OpenBookQA to filter and enrich | Planned |
| 4 | Run B.4 on filtered questions to generate dialogues | Planned |
| 5 | Score with B.5 rubric; drop low-quality dialogues | Planned |
| 6 | Publish combined dataset (translated + generated) | Stretch goal |
| 7 | Fine-tune Qwen2.5-7B or 14B on SocratDataset-EN (LoRA) | Stretch goal |
| 8 | Evaluate using PRR/NDAR/SPR/IAR + B.5 rubric; compare to Table 1 baselines | Stretch goal |

Steps 3–5 can begin in parallel with step 1 since they use different hardware. Steps 7–8 require step 2 to be complete.

---

## 10. Estimated effort

| Task | LLM calls | Est. cost (Claude Sonnet API) | Est. time |
|---|---|---|---|
| B.3 filter/enrich on ARC-Easy (2.3K) | ~2.3K | ~$2–4 | 1–2 hr |
| B.4 dialogue gen on passing questions (~1.1K) | ~1.1K | ~$5–8 | 2–4 hr |
| B.5 quality scoring | ~1.1K | ~$2–3 | 1 hr |
| Same for OpenBookQA (~5K source) | ~10K total | ~$15–25 | 4–8 hr |

All steps are batchable. Using local models (Qwen3.5-27B on the R9700 or Qwen3.5-9B via llama.cpp) reduces cost to near zero at ~4–7× longer wall time.

---

## 11. References

- Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models," EMNLP 2025 Findings — Appendix B.3 (filter prompt), B.4 (dialogue construction prompt), B.5 (evaluation rubric)
- CSQ dataset (Liu et al., 2025) — original Chinese question source
- ARC: Clark et al., 2018
- OpenBookQA: Mihaylov et al., 2018
- SciQ: Welbl et al., 2017
