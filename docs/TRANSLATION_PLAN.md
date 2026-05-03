# SocratDataset English Translation Plan

**CSEN 346 · Santa Clara University**

Goal: produce `ulises-c/SocratDataset-EN` — an English translation of the original Chinese SocratDataset, preserving all structural annotations intact, suitable for training and evaluating an English-language KELE system.

---

## 1. Data provenance

| Layer | Source |
|---|---|
| Original dataset | `references/KELE/SocratDataset.json` — sourced from `yuanpan1020/KELE` on GitHub (MIT license) |
| HF mirror (Chinese) | `ulises-c/SocratDataset` — 6,803 rows, pushed from the reference file |
| Translation target | `ulises-c/SocratDataset-EN` — new HF dataset, same schema, English text fields |

The original dataset was created by Peng et al. (EMNLP 2025) as part of the KELE framework. It covers Chinese elementary school science (grades 1–6, both volumes) structured as Socratic multi-turn teaching dialogues.

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
| Unique action strings | 81 total — ~62 Chinese (LLM-translated once), ~19 state codes / passthrough |

---

## 3. Field translation policy

### 3a. LLM-translate (Chinese → English)

| Field | Location | Notes |
|---|---|---|
| `question` | Row-level | Core problem statement |
| `options` | Row-level | List of 2–4 answer choices |
| `newHint` | Row-level | Short learning clue |
| `newKnowledgePoint` | Row-level | Concept explanation |
| `newAnalyze` | Row-level | Answer analysis |
| `dialogue[].student` | Per-turn | Student utterance |
| `dialogue[].teacher` | Per-turn | Teacher Socratic response — highest-value field |
| `dialogue[].evaluation` | Per-turn | Pedagogical assessment text |

### 3b. Categorical map (no LLM needed)

Pre-translated as hardcoded lookup tables in `translate_dataset.py`:

| Field | Values |
|---|---|
| `mission` | 选择题 → `"multiple_choice"`, 判断题 → `"true_false"` |
| `grade` | 12 values → `"Grade N Vol. V"` format |
| `dialogue[].action` | 62 unique Chinese strings translated in one upfront batch call; state codes (`a1`–`e34`), `"None"`, and empty string pass through unchanged |

### 3c. Pass through unchanged

| Field | Reason |
|---|---|
| `id` | Numeric identifier |
| `chapter` | Float (1.1–4.7) |
| `answer` | Integer index |
| `dialogueRound` | Integer |
| `dialogue[].state` | Cognitive state code — language-agnostic label |

---

## 4. Translation hardware and model

| Setting | Value |
|---|---|
| GPU | AMD Radeon AI PRO R9700 · 32 GB VRAM · RDNA 4 · ROCm |
| **Model** | **Qwen3.5-27B** |
| **Quantization** | **Q4_K_M (~16.5 GB VRAM)** |
| Context window | 262,144 tokens natively |
| Inference stack | vLLM + ROCm |

**Why Qwen3.5-27B Q4_K_M:**
- Uses ~16.5 GB of 32 GB, leaving headroom for KV cache on long batches
- Qwen series leads Chinese↔English translation benchmarks among open models
- Q4_K_M is within ~0.5–1% perplexity of BF16 — effectively lossless for translation
- Released Feb 2026; throughput-optimized (vs. Qwen3.6 which is reasoning-focused — lower throughput for batch work)

**Alternatives considered:**

| Model | Q4_K_M VRAM | Notes |
|---|---|---|
| Qwen3.6-27B-Instruct | ~16.8 GB | Latest (Apr 2026), reasoning-focused — good quality, lower throughput |
| Qwen3.5-35B-A3B-Instruct | ~21 GB | MoE, 3B active — fast, but tighter on VRAM |
| Gemma 4 27B INT8 | ~26 GB | Previous recommendation; Qwen edges it for Chinese |

**Qwen-MT:** Alibaba's `qwen-mt-turbo` is RL-tuned specifically for translation and reportedly competitive with GPT-4.1. It is **API-only** (Alibaba Cloud) — no open weights for local deployment.

**ROCm note:** AMD's "Day 0 support" targets AMD Instinct (MI300X/MI325X, CDNA). The R9700 is RDNA 4. vLLM ROCm support for RDNA 4 is validated through existing project runs; run the smoke test before the full overnight job.

---

## 5. Script: `src/project/translate_dataset.py`

### Configuration block (edit before running)

```python
PUSH_TO_HUB: bool = True                # auto-upload dataset + checkpoints to HF
HF_REPO: str = "ulises-c/SocratDataset-EN"
HF_CHECKPOINT_EVERY: int = 500          # upload checkpoint to HF every N records (0 to disable)
LOCAL_CHECKPOINT_EVERY: int = 50        # save local checkpoint every N records
MODEL: str = "Qwen/Qwen3.5-27B"
BASE_URL: str = "http://localhost:8000/v1"
INPUT_PATH: str = "references/KELE/SocratDataset.json"
OUTPUT_PATH: str = "references/KELE/SocratDataset-EN.json"
CHECKPOINT_PATH: str = "references/KELE/translate_checkpoint.json"
```

### CLI usage

```bash
# Auth — run once before any HF upload:
hf login

# Smoke test — 10 records, verify JSON parse rate and translation quality:
python -m project.translate_dataset --smoke-test 10

# Full overnight run (checkpoints + final push go to HF automatically):
python -m project.translate_dataset

# If the machine died — restore checkpoint from HF, then continue:
python -m project.translate_dataset --restore-from-hub

# Skip all HF uploads for this run (PUSH_TO_HUB=True stays in the file):
python -m project.translate_dataset --no-push

# Override model or server:
python -m project.translate_dataset --model Qwen/Qwen3.6-27B-Instruct --base-url http://localhost:8001/v1
```

### Checkpoint system

Two-tier checkpoint design:

| Tier | Frequency | Location | Purpose |
|---|---|---|---|
| **Local** | Every 50 records | `references/KELE/translate_checkpoint.json` | Fast recovery from script crash / Ctrl-C |
| **HuggingFace** | Every 500 records | `ulises-c/SocratDataset-EN/translate_checkpoint.json` | Recovery if the whole machine dies overnight |

The HF checkpoint uploads the same JSON (translated records so far + action cache) as a repo file via `huggingface_hub.upload_file()`. To resume after a machine failure:

```bash
python -m project.translate_dataset --restore-from-hub
```

This downloads the HF checkpoint to the local path before the main loop starts, then picks up where it left off.

### HF upload summary

| Event | What is uploaded | How |
|---|---|---|
| Every 500 records | `translate_checkpoint.json` (partial results) | `upload_file()` to dataset repo |
| End of run | Full translated dataset as Parquet | `Dataset.push_to_hub()` |

Auth for all uploads is handled automatically from `hf login` (stored token) or `HF_TOKEN` env var — no token in code.

---

## 6. Output format

Published to HF as `ulises-c/SocratDataset-EN` in Parquet. A `translation_meta` field is added at the row level:

```json
{
  "translation_meta": {
    "model": "Qwen/Qwen3.5-27B",
    "translated_at": "2026-xx-xxTxx:xx:xxZ"
  }
}
```

The Chinese original remains at `ulises-c/SocratDataset` — the English dataset is additive.

---

## 7. Batching strategy

Each row is translated as a single structured prompt containing all LLM-translate fields. The model returns a JSON object with translated fields and identical structure.

**Batch size:** 1 record per request initially (conservative). The 262K context window easily fits 5–10 records; experiment after confirming baseline quality on the smoke test.

**Estimated throughput:** Qwen3.5-27B Q4_K_M on the R9700 at ~400–700 tok/s. Each record produces ~400–800 translated tokens. At 550 tok/s: ~6,800 × 600 tokens / 550 tok/s ≈ **~2–4 hours** for the full dataset.

---

## 8. Quality validation

Before treating the dataset as final, run a validation pass on a random 5% sample (~340 records):

| Check | Method |
|---|---|
| **JSON parse rate** | All 6,803 rows must produce valid JSON; failures are logged to stderr |
| **Field completeness** | Every LLM-translate field must be non-empty |
| **State label preservation** | Assert `dialogue[].state` values are unchanged |
| **Option count preservation** | Assert `len(options_en) == len(options_zh)` for every record |
| **Back-translation consistency** | Translate English → Chinese; compute embedding cosine similarity; flag rows below ~0.85 |
| **Human spot-check** | Manually review 20–30 records across grade levels and mission types |

---

## 9. Prerequisites before running

1. Confirm vLLM + ROCm loads Qwen3.5-27B on the R9700 (smoke test first)
2. Download `Qwen/Qwen3.5-27B` weights (GGUF Q4_K_M or GPTQ depending on vLLM backend)
3. Run `hf login` so checkpoint uploads and final push work
4. Run 10-record smoke test; verify JSON parse rate and translation quality before committing to the full overnight run

---

## 10. Intended downstream uses

| Use case | How this dataset helps |
|---|---|
| English KELE replication | Run the KELE pipeline with an English teacher model on a translated test set |
| Cross-lingual transfer | Train the hierarchical BERT classifier on English BERT; compare state accuracy vs. Chinese BERT baseline |
| HF contribution | Standalone research artifact — the only annotated English Socratic science tutoring dataset with 34-strategy / 30-state labels |
| Augmented training | Merge with MathDial or other English tutoring datasets for a combined fine-tune |

---

## 11. Sequencing

This is a preprocessing task with no dependency on ongoing KELE evaluation runs. It can run overnight on the R9700 while the 5090 handles other experiments. The translated dataset is not required for any current evaluation milestones (June 4 deadline) — it is a stretch goal and HF contribution.
