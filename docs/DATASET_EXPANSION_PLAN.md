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

## 5. Recommended sequencing

| Step | Action | Status |
|---|---|---|
| 1 | Translate SocratDataset (Chinese → English) | In progress — ~9 hr remaining |
| 2 | Publish `ulises-c/SocratDataset-EN` to HuggingFace | Blocked on step 1 |
| 3 | Run B.3 on ARC-Easy + OpenBookQA to filter and enrich | Planned |
| 4 | Run B.4 on filtered questions to generate dialogues | Planned |
| 5 | Score with B.5 rubric; drop low-quality dialogues | Planned |
| 6 | Publish combined dataset (translated + generated) | Stretch goal |

Steps 3–5 can begin in parallel with step 1 since they use different hardware.

---

## 6. Estimated effort

| Task | LLM calls | Est. cost (Claude Sonnet API) | Est. time |
|---|---|---|---|
| B.3 filter/enrich on ARC-Easy (2.3K) | ~2.3K | ~$2–4 | 1–2 hr |
| B.4 dialogue gen on passing questions (~1.1K) | ~1.1K | ~$5–8 | 2–4 hr |
| B.5 quality scoring | ~1.1K | ~$2–3 | 1 hr |
| Same for OpenBookQA (~5K source) | ~10K total | ~$15–25 | 4–8 hr |

All steps are batchable. Using local models (Qwen3.5-27B on the R9700 or Qwen3.5-9B via llama.cpp) reduces cost to near zero at ~4–7× longer wall time.

---

## 7. References

- Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models," EMNLP 2025 Findings — Appendix B.3 (filter prompt), B.4 (dialogue construction prompt), B.5 (evaluation rubric)
- CSQ dataset (Liu et al., 2025) — original Chinese question source
- ARC: Clark et al., 2018
- OpenBookQA: Mihaylov et al., 2018
- SciQ: Welbl et al., 2017
