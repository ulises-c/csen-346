# SocratDataset English Translation Plan

**CSEN 346 · Santa Clara University**

Goal: produce `ulises-c/SocratDataset-en` — an English translation of the original Chinese SocratDataset, preserving all structural annotations intact, suitable for training and evaluating an English-language KELE system.

---

## 1. Data provenance

| Layer | Source |
|---|---|
| Original dataset | `references/KELE/SocratDataset.json` — sourced from `yuanpan1020/KELE` on GitHub (MIT license) |
| HF mirror (Chinese) | `ulises-c/SocratDataset` — 6,803 rows, pushed from the reference file |
| Translation target | `ulises-c/SocratDataset-en` — new HF dataset, same schema, English text fields |

The original dataset was created by Peng et al. (EMNLP 2025) as part of the KELE framework. It covers Chinese elementary school science (grades 1–6, both volumes) structured as Socratic multi-turn teaching dialogues. The raw source is the `SocratDataset.json` file in the KELE GitHub repository, already present locally at `references/KELE/SocratDataset.json`.

---

## 2. Dataset profile

| Property | Value |
|---|---|
| Total records | 6,803 |
| Total dialogue turns | ~42,000+ |
| Dialogue rounds per record | 5–12 (median: 6) |
| Mission types | 选择题 (multiple choice): 4,312 · 判断题 (true/false): 2,491 |
| Grade levels | Grades 1–6, volumes 1–2 (12 distinct values) |
| Unique state labels | 34 (a1, b2–b7, c8–c29, d30–d33, e34) |
| Unique action strings | ~15 (highly repetitive across 42,000+ turns) |
| Chinese chars to translate | ~5.4M |
| Estimated translation tokens | ~7M (chars × 1.3) |

---

## 3. Field translation policy

### 3a. LLM-translate (Chinese → English)

These fields contain natural language that requires semantic translation:

| Field | Location | Notes |
|---|---|---|
| `question` | Row-level | Core problem statement; avg 28 chars, max 178 |
| `options` | Row-level | List of 2–4 answer choices; translate each item |
| `newHint` | Row-level | Short learning clue (9–37 chars) |
| `newKnowledgePoint` | Row-level | Concept explanation (12–167 chars) |
| `newAnalyze` | Row-level | Answer analysis (38–314 chars) |
| `dialogue[].student` | Per-turn | Student utterance; includes question + options in first turn |
| `dialogue[].teacher` | Per-turn | Teacher Socratic response — highest-value field |
| `dialogue[].evaluation` | Per-turn | Pedagogical assessment text; moderately repetitive |

### 3b. Categorical map (no LLM needed)

Pre-translate as lookup tables before the main job runs:

| Field | Chinese values | English mapping |
|---|---|---|
| `mission` | 选择题 | `"multiple_choice"` |
| | 判断题 | `"true_false"` |
| `grade` | 1小学一年级上册 … 6小学六年级下册 | `"Grade 1 Vol. 1"` … `"Grade 6 Vol. 2"` |
| `dialogue[].action` | ~15 unique strings | Pre-translate once; apply via lookup for all 42k turns |

The `action` field is the most important optimization: ~15 strings repeated across 42,000+ turns. Translating them once and looking up the result eliminates ~99% of action-field LLM calls.

### 3c. Pass through unchanged

| Field | Reason |
|---|---|
| `id` | Numeric identifier |
| `chapter` | Float (1.1–4.7) |
| `answer` | Integer index (1-indexed) |
| `dialogueRound` | Integer |
| `dialogue[].state` | Cognitive state code (`a1`, `b2`, etc.) — language-agnostic label; do not translate |

---

## 4. Translation hardware and model

| Setting | Value |
|---|---|
| GPU | AMD Radeon AI PRO R9700 · 32 GB VRAM · RDNA 4 · ROCm |
| Model | Gemma 4 26B MoE |
| Quantization | INT8 (~26 GB VRAM, best quality available on AMD) |
| Context window | Up to 256K tokens (cap batch size to stay within ~32K for KV headroom) |
| Inference stack | vLLM + ROCm (same stack already validated for SocratTeachLLM on this GPU) |
| Run mode | Standalone batch job — no other models loaded |

**Why 26B MoE INT8 over 31B INT4:** INT8 is within ~0.5% perplexity of BF16 (effectively lossless for translation). The 26B MoE at INT8 preserves higher numeric precision than 31B GPTQ/AWQ at INT4. The MoE's 4B active parameters also give faster per-token throughput than a 31B dense model. NVFP4 is NVIDIA Blackwell-specific and is not available on AMD ROCm.

---

## 5. Batching strategy

Each row is translated as a single structured prompt containing all LLM-translate fields for that record. The model is asked to return a JSON object with the translated fields, leaving structure and keys identical.

**Per-row prompt contents:**
- System instruction: role, task, language direction, output format
- One-shot example (hardcoded, not from dataset)
- The record's fields as a JSON object

**Batch size:** 1 record per request initially. If throughput is acceptable, experiment with 5–10 records per request (the 256K window easily fits them). Multi-record batching risks partial JSON failures; start conservative.

**Action field:** resolved via lookup table before the LLM job; the LLM never sees action fields.

**Categorical fields:** resolved via lookup table before the LLM job.

**Estimated throughput:** Gemma 4 26B INT8 on the R9700 at ~500–800 tok/s output. Each record produces ~400–800 translated tokens. At 600 tok/s average: ~6,800 records × 600 tokens / 600 tok/s ≈ **~2–4 hours** for the full dataset. Overnight batch job.

---

## 6. Output format

The translated dataset will be published to HF as `ulises-c/SocratDataset-en` with the same Parquet schema as `ulises-c/SocratDataset`. A `translation_meta` field will be added at the row level:

```json
{
  "translation_meta": {
    "model": "google/gemma-4-26b-it",
    "quantization": "int8",
    "hardware": "AMD R9700 32GB ROCm",
    "translated_at": "2026-xx-xx"
  }
}
```

The Chinese original dataset remains at `ulises-c/SocratDataset` — the English dataset is additive, not a replacement.

---

## 7. Quality validation

Before pushing to HF, run a validation pass on a random 5% sample (~340 records):

| Check | Method |
|---|---|
| **Back-translation consistency** | Translate English → Chinese with the same model; compute embedding cosine similarity against original Chinese. Flag rows below threshold (e.g., 0.85). |
| **JSON parse rate** | All 6,803 rows must produce valid JSON. Track and log any failures for manual review. |
| **Field completeness** | Every LLM-translate field must be non-empty in the output. |
| **State label preservation** | Assert that `dialogue[].state` values are unchanged in the output. |
| **Option count preservation** | Assert that `len(options_en) == len(options_zh)` for every record. |
| **Human spot-check** | Manually review 20–30 records across grade levels and mission types for translation naturalness and pedagogical fidelity. |

---

## 8. Known limitations and risks

| Risk | Mitigation |
|---|---|
| **Pedagogical style** | A translated Chinese math teacher will not sound like a native English math tutor. The translated dataset is useful for framework validation and cross-lingual transfer experiments, not as a replacement for native English tutoring data. |
| **Curriculum specificity** | Some problems reference Chinese elementary school science curriculum framing. Translation preserves the text but not the cultural grounding. |
| **JSON output failures** | LLMs occasionally malform JSON on long outputs. Implement retry logic with a stricter prompt on failure; fall back to per-field translation for problematic records. |
| **ROCm + vLLM + Gemma 4 26B INT8** | Gemma 4 was released April 2026; verify vLLM ROCm support for the 26B MoE architecture before the main run. Run a 10-record smoke test first. |
| **Domain shift** | The English translation will have different embedding-space properties than natively English datasets like MathDial. Label this clearly in the HF dataset card. |

---

## 9. Intended downstream uses

| Use case | How this dataset helps |
|---|---|
| English KELE replication | Run the full KELE pipeline with an English teacher model (Llama 3, Mistral, etc.) on a translated test set |
| Cross-lingual transfer | Train the hierarchical BERT classifier (Improvement #2) on English BERT; compare state accuracy against the Chinese BERT baseline |
| HF contribution | A standalone research artifact — the only annotated English Socratic math tutoring dataset with 34-strategy / 30-state labels |
| Augmented training | Merge with MathDial or other English tutoring datasets for a combined English tutoring fine-tune |

---

## 10. Sequencing relative to other work

This is a preprocessing task with no dependency on ongoing KELE evaluation runs. It can run overnight on the R9700 while the 5090 handles other experiments. The translated dataset is not required for any current evaluation milestones (June 4 deadline) — it is a stretch goal and HF contribution.

**Prerequisites before running:**
1. Confirm vLLM ROCm support for Gemma 4 26B MoE (check vLLM release notes / GitHub issues)
2. Download Gemma 4 26B MoE INT8 checkpoint to local storage
3. Pre-build the categorical lookup tables (grade, mission, action) and verify completeness
4. Run 10-record smoke test; check JSON parse rate and translation quality before committing to full run
