# RTX 5090 Setup Runbook

Step-by-step guide to get both models downloaded, served, and evaluated on the RTX 5090 32GB rig.

---

## Prerequisites

```bash
# System
nvidia-smi                  # Confirm CUDA driver is working
python --version            # Need 3.10+

# Core packages
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install vllm transformers accelerate huggingface_hub
pip install rouge-score sacrebleu openai
pip install wandb                          # Training metrics & dashboards
pip install tqdm rich                      # Progress bars & rich console output
```

---

## Step 1 — Download Models

Download both models upfront so we're not bottlenecked by network during experiments.

```bash
# Create a local model cache
export HF_HOME=~/hf_models
mkdir -p $HF_HOME

# Model 1: SocratTeachLLM (9.4B, ~19GB, BF16)
huggingface-cli download yuanpan/SocratTeachLLM --local-dir ~/hf_models/SocratTeachLLM

# Model 2: Gemma-4-31B (pick ONE quantization)
# Option A: Full weights (~62GB) — won't fit in 32GB VRAM, need quantization at load time
huggingface-cli download google/gemma-4-31B --local-dir ~/hf_models/gemma-4-31B

# Option B (recommended): Pre-quantized GGUF for faster startup
# Check for community Q4 quants on HF — search "gemma-4-31B GGUF" or "gemma-4-31B AWQ"
# Example (if available):
# huggingface-cli download <user>/gemma-4-31B-AWQ --local-dir ~/hf_models/gemma-4-31B-AWQ
```

### Verify downloads

```bash
ls -lh ~/hf_models/SocratTeachLLM/    # Expect ~19GB across 4 shards
ls -lh ~/hf_models/gemma-4-31B/       # Expect ~62GB full or ~17GB Q4
```

---

## Step 2 — Serve SocratTeachLLM (Baseline)

vLLM gives us an OpenAI-compatible API that the existing KELE code can hit directly.

```bash
# Serve SocratTeachLLM on port 8001
vllm serve ~/hf_models/SocratTeachLLM \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  2>&1 | tee logs/vllm_socrat.log
```

**Important:** SocratTeachLLM uses custom code (`trust-remote-code` is required).

### Test the endpoint

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SocratTeachLLM",
    "messages": [{"role": "user", "content": "Hello, can you help me with a science question?"}],
    "max_tokens": 200
  }'
```

---

## Step 3 — Serve Gemma-4-31B

After baseline experiments are done, swap to Gemma.

```bash
# Kill the SocratTeachLLM server first (or use a different port)

# Option A: vLLM with on-the-fly quantization
vllm serve ~/hf_models/gemma-4-31B \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --quantization awq \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee logs/vllm_gemma.log

# Option B: If using pre-quantized AWQ weights
vllm serve ~/hf_models/gemma-4-31B-AWQ \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee logs/vllm_gemma.log
```

### VRAM budget

| Model | Precision | VRAM Usage | Fits in 32GB? |
| --- | --- | --- | --- |
| SocratTeachLLM 9.4B | BF16 | ~19GB | Yes, comfortably |
| Gemma-4-31B | Q4 (AWQ) | ~17GB | Yes |
| Gemma-4-31B | Q8 | ~33GB | Tight — may OOM with long contexts |
| Gemma-4-31B | BF16 | ~62GB | No |

---

## Step 4 — Run the KELE System

Both models get served on the same OpenAI-compatible endpoint, so the code doesn't change — only the model name in the config.

```bash
# .env file (create in project root)
CONSULTANT_API_KEY=sk-xxxxx              # OpenAI API key for GPT-4o
CONSULTANT_BASE_URL=https://api.openai.com/v1
CONSULTANT_MODEL_NAME=gpt-4o

TEACHER_API_KEY=not-needed               # vLLM doesn't require a real key
TEACHER_BASE_URL=http://localhost:8001/v1
TEACHER_MODEL_NAME=SocratTeachLLM        # Swap to gemma-4-31B for Run 2
```

---

## Step 5 — Evaluation Pipeline with Metrics Export

### Progress tracking during evaluation runs

```
results/
├── baseline/
│   ├── run_config.json          # Model name, quantization, timestamp, hyperparams
│   ├── dialogues/               # Raw dialogue outputs (one JSON per dialogue)
│   ├── metrics_summary.json     # ROUGE-1/2/L, BLEU-4, PRR, NDAR, SPR, IAR
│   ├── multi_turn_scores.json   # Guidance, Logicality, Flexibility, Repetitiveness, Clarity
│   └── progress.log             # Live progress: completed/total, ETA, tokens/sec
├── gemma/
│   ├── (same structure)
│   └── ...
└── comparison.json              # Side-by-side comparison of both runs
```

### Estimated time to completion (ETC) tracking

Every evaluation script should print and log:
```
[182/680 dialogues] 26.8% complete | 42.3 tok/s | ETA: 2h 14m | Elapsed: 0h 49m
```

### Metrics to capture per run

| Category | Metrics | Method |
| --- | --- | --- |
| Text overlap | ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4 | `rouge-score`, `sacrebleu` |
| Single-turn (GPT-4o judge) | PRR, NDAR, SPR, IAR | Binary yes/no per turn |
| Multi-turn (GPT-4o judge) | Guidance, Logicality, Flexibility, Repetitiveness, Clarity | 1–5 scale |
| System | tokens/sec, total wall time, VRAM peak, GPU utilization | `nvidia-smi`, logged per batch |

### Weights & Biases (optional but recommended)

```bash
wandb login
# Then in the eval script:
# wandb.init(project="kele-eval", name="baseline-socratteachllm")
# wandb.log({"rouge1": score, "dialogues_completed": n, ...})
```

This gives us live dashboards, automatic ETA, and run comparisons without building custom tooling.

---

## Step 6 — Phase 4: BERT Classifier Training

This trains the replacement consultant agent.

```bash
# Additional dependencies
pip install datasets scikit-learn

# Training (small model, fast even without wandb)
python src/project/consultant_classifier.py \
  --data resources/KELE/SocratDataset.json \
  --model distilbert-base-uncased \
  --epochs 5 \
  --batch-size 32 \
  --lr 2e-5 \
  --output results/classifier/ \
  2>&1 | tee logs/classifier_training.log
```

### Expected training time

| Model | Epochs | RTX 5090 |
| --- | --- | --- |
| distilbert-base | 5 | ~5–10 min |
| bert-base-uncased | 5 | ~8–15 min |

---

## Quick Reference — Command Cheatsheet

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Start SocratTeachLLM server
vllm serve ~/hf_models/SocratTeachLLM --port 8001 --dtype bfloat16 --trust-remote-code

# Start Gemma server
vllm serve ~/hf_models/gemma-4-31B-AWQ --port 8001 --dtype float16

# Run baseline eval
python src/project/evaluate.py --model socratteachllm --output results/baseline/

# Run gemma eval
python src/project/evaluate.py --model gemma --output results/gemma/

# Compare results
python src/project/compare_results.py results/baseline/ results/gemma/
```

---

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `trust_remote_code` error | Add `--trust-remote-code` flag to vLLM |
| OOM on Gemma-4-31B | Use AWQ/Q4 quantization, reduce `--max-model-len` to 2048 |
| vLLM won't load ChatGLM | May need `--tokenizer-mode slow` for custom tokenizers |
| Slow generation | Check `--gpu-memory-utilization` isn't too low, confirm no CPU offload |
| Chinese output from SocratTeachLLM | Model was trained on Chinese data — may need Chinese prompts for teacher agent |
