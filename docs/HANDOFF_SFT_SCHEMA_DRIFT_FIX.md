# Handoff — Stage 2b SFT schema drift fix + R9700 training in progress

**Branch:** `feat/gfx1201-rdna4-qlora-fla-training` (PR #101)  
**Last commit:** `d04bc83`  
**Date:** 2026-05-30

---

## TL;DR

The 5090-trained Gemma 4 31B Stage 2b adapter had output collapse (never emits
EOS, runs to context limit). Root cause was a train/serve schema mismatch, not
model quality. The schema is now fixed. Training is running on the R9700.

---

## Root cause (from 5090 run)

The model was healthy — training-shape prompts produced clean 20-token responses.
Inference-shape prompts caused 2048-token repetition collapse. Decisive A/B:

| | Training (old `socrat-zh/en`) | Inference (`socrates_teacher()`) |
|---|---|---|
| System prompt | 1-line + problem context | 7-line rules block, no problem |
| Message structure | Multi-turn N pairs | Single user message |
| History | HF message turns | `学生:/老师:` text blob |
| Current input label | None | `当前学生输入:` |

See: `docs/EXPERIMENT_LOG.md` (2026-05-28 PM), PR #94 commit `eff214f`.

---

## What was fixed this session

### 1. New SFT data sources (`src/project/dataset.py`)

`socrat-zh-sft` and `socrat-en-sft` replace `socrat-zh`/`socrat-en` for training.
They produce one `(system, user, asst)` triple per dialogue turn in the exact
format `socrates_teacher()` sends at inference.

- System: 7-line rules block (copied from `socratic_teaching_system.py:404-415`)
- User: `\n历史对话记录:\n{学生/老师 pairs}\n\n当前学生输入: ...\n\n苏格拉底教学顾问评估结果: 学生处于 {state} 状态\n苏格拉底教学顾问建议的操作: {action}\n`
- **Render-diff verified MATCH: True** (byte-identical to real inference messages)
- Split is at **dialogue level** — all turns of a dialogue stay in same partition
- Old `socrat-zh`/`socrat-en` multi-turn loaders untouched (backward compat)
- 77,202 train records / 8,578 eval records

### 2. `quantization_config=None` crash fix (`scripts/train_sft.py`)

Passing explicit `None` to `from_pretrained` caused `auto_factory.py` to
overwrite the unsloth checkpoint's `config.json` quantization config with `None`
→ `supports_quant_method` crash. Fix: only include `quantization_config` in
`_load_kwargs` when it is not None.

### 3. W&B integration (`scripts/train_sft.py`, `pyproject.toml`)

- `wandb>=0.18.0` is a core dep (not an extra)
- `_check_wandb()` calls `wandb.login(relogin=False)` — warns and disables if
  unauthenticated, no flag needed
- `report_to=["wandb"]`, `run_name` defaults to output-dir basename
- Credentials present in `~/.netrc` → tracking is live
- W&B project: `csen346-sft`, run: `sft-stage2-gemma4-31b`

---

## Current training state

```
Model:   unsloth/gemma-4-31B-it-unsloth-bnb-4bit  (pre-quantized NF4, ~19 GB)
Sources: socrat-zh-sft, socrat-en-sft (77,202 per-turn records)
Config:  configs/train-sft-stage2-gemma4-31b.env
Output:  outputs/sft-stage2-gemma4-31b/
Steps:   2,298 total (3 epochs × 766 steps/epoch)
Pace:    ~84 s/step on R9700 gfx1201 QLoRA
ETA:     ~55 h from start (~2026-06-01 23:00)
Saves:   every 100 steps (~2.4 h rollback granularity)
```

Monitor: `tail -f outputs/sft-stage2-gemma4-31b/train.log`  
W&B: https://wandb.ai/uchavarria-santa-clara-university/csen346-sft/runs/thb2cnzq

**Launch command (to restart if killed):**
```bash
nohup env TORCH_USE_HIPBLASLT=0 \
  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,expandable_segments:True \
  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  TRAIN_PREQ=true \
  uv run --no-sync python scripts/train_sft.py \
  --config configs/train-sft-stage2-gemma4-31b.env \
  > outputs/sft-stage2-gemma4-31b/train.log 2>&1 &
```

Auto-resumes from latest checkpoint if `outputs/sft-stage2-gemma4-31b/checkpoint-*` exists.

---

## Blocking gate before trusting the full run

After the **first checkpoint** (~100 steps, ~2.4 h in), load it and probe with
an inference-shape prompt. The model must emit clean EOS — not a repetition loop.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-4-31B-it-unsloth-bnb-4bit", device_map={"": 0})
model = PeftModel.from_pretrained(base, "outputs/sft-stage2-gemma4-31b/checkpoint-100")
tok = AutoTokenizer.from_pretrained("outputs/sft-stage2-gemma4-31b/checkpoint-100")

# Inference-shape prompt (must produce clean output, not a repetition loop)
msgs = [
    {"role": "system", "content": "<7-line rules block>"},
    {"role": "user", "content": "\n历史对话记录:\n\n\n当前学生输入: 植物的种子在哪里？\n\n苏格拉底教学顾问评估结果: 学生处于 a1 状态\n苏格拉底教学顾问建议的操作: 生成一个问题\n"},
]
ids = tok.apply_chat_template(msgs, return_tensors="pt").to("cuda")
out = model.generate(ids, max_new_tokens=100, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:]))
```

**Pass:** single coherent Socratic question, EOS within 50 tokens.  
**Fail:** repetition of any phrase, or `max_new_tokens` exhausted.

---

## Known residual issues (deferred)

1. **Evaluation string drift** — training uses `学生处于 {state} 状态`; inference
   injects the consultant's free-form reasoning paragraph. The A/B test confirmed
   the terse form is sufficient for EOS. Address when a re-annotated dataset
   is available.

2. **`_build_inference_user_message` divergence risk** — reimplements
   `get_formatted_history()` from `socratic_teaching_system.py`. Render-diff is
   the mandatory gate before changing either function.

3. **No tests for new loaders** — `load_socrat_zh_sft` / `load_socrat_en_sft`
   and `_build_inference_user_message` have no unit tests. Add before any refactor.

---

## After training completes

1. **Load checkpoint smoke test** (see blocking gate above)
2. **Merge LoRA** → BF16 HF checkpoint: `python scripts/merge_lora_gemma4_sft.py`
3. **Convert to GGUF**: `bash scripts/convert_gemma4_sft_to_gguf.sh`
4. **Serve and eval**: replace `GEMMA4_31B_WEIGHT_FILE` with the new GGUF in
   `scripts/serve_gemma4_31b_q5.sh`, run `make eval-gemma4-31b-full`
5. **Path A from PR #94 becomes live**: eval numbers (likely bad per §2.8
   outcome matrix) are the paper's negative-result contribution.

---

## Files changed this session

| File | Change |
|---|---|
| `src/project/dataset.py` | `socrat-zh-sft` / `socrat-en-sft` sources; `_TEACHER_INFERENCE_SYSTEM`; `_build_inference_user_message`; dialogue-level split fix |
| `scripts/train_sft.py` | `_check_wandb()`; `_load_kwargs` conditional build; W&B `report_to` / `run_name` |
| `configs/train-sft-stage2-gemma4-31b.env` | `TRAIN_SOURCES=socrat-zh-sft,socrat-en-sft`; W&B stanza |
| `pyproject.toml` | `wandb>=0.18.0` core dep |
| `.gitignore` | `wandb/` artifacts |
| `.env.example` | `WANDB_API_KEY` section |

PR comments: #4585166874 (schema fix summary), #4585192824 (split fix patch)
