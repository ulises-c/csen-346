# RTX 3070 Setup Runbook

Adapted from `RTX5090_SETUP.md` for RTX 3070 (8 GB VRAM).

---

## Key Differences vs. RTX 5090

| | RTX 5090 (32 GB) | RTX 3070 (8 GB) |
|---|---|---|
| SocratTeachLLM precision | BF16 | NF4 4-bit (bitsandbytes) |
| SocratTeachLLM VRAM | ~19 GB | ~5–6 GB |
| Max context length | 4096 tokens | 1536 tokens |
| Throughput | 80–120 tok/s | 12–20 tok/s |
| Full eval (680 dialogues) | 1.5–3 hrs | 8–16 hrs |
| Run 2 model | Gemma-4-31B Q4 (17 GB) | Gemma-3-4B-IT NF4 (~2.5 GB) |
| Serving layer | vLLM | transformers + bitsandbytes + FastAPI |
| BERT classifier training | 3–8 min | 10–25 min |

**Why not vLLM?** vLLM's CUDA graphs and memory pools add ~1–2 GB overhead on top of the
model weights, which pushes a 9.4B NF4 model over the 8 GB limit. The `transformers` +
`bitsandbytes` path reaches the same result at lower VRAM cost.

---

## Prerequisites

### Confirmed working on this machine
```
GPU:    NVIDIA RTX 3070  (8 GB VRAM)
Driver: 595.45.04   CUDA runtime: 13.2
OS:     CachyOS (Arch-based)
Python: 3.14.3
Poetry: 2.3.4
```

---

## Step 0 — Install the Environment

### 1. Install non-torch packages via Poetry

```bash
cd /path/to/csen-346
poetry install --no-root
# This will FAIL on torch/triton — that is expected.
# All other packages install correctly.
```

### 2. Install the CUDA-enabled torch wheel manually

Poetry's resolver cannot reconcile the +cu126 local version identifier with the
transitive torch dependency from transformers. Install torch directly into the
poetry venv instead:

```bash
poetry run pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  "torch>=2.10.0"
```

### 3. Verify

```bash
poetry run python -c "
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
"
# Expected:
#   torch: 2.11.0+cu126
#   CUDA: True
#   GPU: NVIDIA GeForce RTX 3070
```

### 4. Copy and fill in .env

```bash
cp .env.example .env
# Edit .env — at minimum set CONSULTANT_API_KEY to your OpenAI API key
```

---

## Step 1 — Download SocratTeachLLM

```bash
mkdir -p ~/hf_models
poetry run huggingface-cli download yuanpan/SocratTeachLLM \
  --local-dir ~/hf_models/SocratTeachLLM
```

Expected: ~19 GB download (BF16 weights). We load them in NF4 at runtime so the
on-disk weights don't need to be re-quantized.

---

## Step 2 — Start the Teacher Model Server

The server loads SocratTeachLLM in **NF4 4-bit** via bitsandbytes and exposes an
OpenAI-compatible `/v1/chat/completions` endpoint on port 8001.

```bash
TEACHER_LOCAL_PATH=~/hf_models/SocratTeachLLM \
  poetry run python -m src.project.serve_teacher
```

Expected VRAM usage after loading: **~5.5–6.2 GB**
Expected generation speed: **12–20 tok/s**

### Test the endpoint

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SocratTeachLLM",
    "messages": [{"role": "user", "content": "What is photosynthesis?"}],
    "max_tokens": 200
  }'
```

---

## Step 3 — Run the KELE System

With the server running on port 8001 and `.env` configured:

```bash
poetry run python -m src.project.kele
```

---

## Step 4 — (Optional) Run 2: Gemma-3-4B-IT

Gemma-4-31B won't fit on 8 GB at any quantization level.
Use Gemma-3-4B-IT (4B parameters) as a comparable open-source baseline instead.

```bash
poetry run huggingface-cli download google/gemma-3-4b-it \
  --local-dir ~/hf_models/gemma-3-4b-it
```

Then serve it:

```bash
TEACHER_LOCAL_PATH=~/hf_models/gemma-3-4b-it \
  TEACHER_MODEL_NAME=gemma-3-4b-it \
  poetry run python -m src.project.serve_teacher
```

Update `.env` → `TEACHER_MODEL_NAME=gemma-3-4b-it` and re-run KELE for Run 2.

Expected VRAM: ~2.5–3.0 GB at NF4. Expected speed: ~30–50 tok/s.

---

## Step 5 — Evaluation Pipeline

Same as RTX5090_SETUP.md steps 4–5. The only difference is throughput and wall time.

```
results/
├── baseline/        # SocratTeachLLM NF4 run
├── gemma/           # Gemma-3-4B-IT NF4 run
└── comparison.json
```

---

## Step 6 — BERT Classifier Training (Phase 4)

No changes needed. DistilBERT fits easily in 8 GB.

```bash
poetry run python src/project/consultant_classifier.py \
  --data resources/KELE/SocratDataset.json \
  --model distilbert-base-uncased \
  --epochs 5 \
  --batch-size 16 \
  --lr 2e-5 \
  --output results/classifier/
```

Use `--batch-size 16` instead of 32 (RTX 5090 setting) to stay inside the VRAM budget.

Expected training time: **10–25 min** (vs. 5–10 min on RTX 5090).

---

## VRAM Budget Reference

| Task | VRAM |
|---|---|
| Desktop idle | ~1.5 GB |
| SocratTeachLLM NF4 loaded | ~5.5–6.2 GB |
| SocratTeachLLM NF4 + active inference | ~6.5–7.5 GB |
| Gemma-3-4B-IT NF4 loaded | ~2.5–3.0 GB |
| DistilBERT fine-tuning (batch=16) | ~3–4 GB |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `trust_remote_code` error | Already enabled by default in `serve_teacher.py` |
| OOM during inference | Reduce `MAX_NEW_TOKENS` in `.env` (default 512 → try 256) |
| OOM loading model | Check desktop VRAM; close GPU-heavy apps; ensure NF4 is active |
| Slow inference | Expected — 12–20 tok/s is normal for NF4 on 3070; full eval ~8–16 hrs |
| Chinese output from SocratTeachLLM | Model trained on Chinese data; the existing KELE prompts in English should still work, but responses may be Chinese |
| `torch install failed` in poetry | Normal — see Step 0 above; use manual pip install step |
