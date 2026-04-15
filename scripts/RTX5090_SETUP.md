# RTX 5090 Setup Runbook

Step-by-step guide to get all models downloaded, served, and evaluated on the RTX 5090 32GB rig.

**Architecture:** Fully local, zero paid APIs. The consultant (Qwen3.5-9B) and teacher (SocratTeachLLM) both run on the same GPU via vLLM on separate ports.

---

## Prerequisites

```bash
# System
nvidia-smi                  # Confirm CUDA driver is working

# Python 3.12 via pyenv (Arch ships 3.14 which is too new for vLLM)
pyenv install 3.12
cd ~/Documents/scu/CSEN-346/csen-346
~/.pyenv/versions/3.12.*/bin/python -m venv .venv
source .venv/bin/activate

# Core packages (inside venv)
pip install openai vllm transformers accelerate huggingface_hub
pip install rouge-score sacrebleu
pip install wandb                          # Training metrics & dashboards
pip install tqdm rich                      # Progress bars & rich console output
```

---

## Step 1 — Download Models

Download all models upfront so we're not bottlenecked by network during experiments.

```bash
mkdir -p ~/hf_models

# Model 1: SocratTeachLLM (9.4B, GLM4-9B fine-tune, ~19GB BF16) — teacher agent
huggingface-cli download yuanpan/SocratTeachLLM --local-dir ~/hf_models/SocratTeachLLM

# Model 2: Qwen3.5-9B (~17GB BF16, ~5GB Q4) — consultant agent (replaces GPT-4)
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/hf_models/Qwen3.5-9B

# Model 3 (Phase 1 Run 2): Gemma-4-31B — extension experiment
# huggingface-cli download google/gemma-4-31B --local-dir ~/hf_models/gemma-4-31B
```

### Verify downloads

```bash
ls -lh ~/hf_models/SocratTeachLLM/    # Expect ~19GB across 4 shards
ls -lh ~/hf_models/Qwen3.5-9B/        # Expect ~17GB BF16
```

---

## Step 2 — Serve Both Models

Both models run simultaneously on the same GPU via vLLM on separate ports. vLLM gives us OpenAI-compatible APIs that the KELE code hits directly.

```bash
# Option A: One command to start both
./scripts/serve_both.sh

# Option B: Start individually in separate terminals
./scripts/serve_socratteachllm.sh   # Teacher on port 8001
./scripts/serve_consultant.sh       # Consultant on port 8002
```

### Test the endpoints

```bash
curl http://localhost:8001/v1/models   # Should show SocratTeachLLM
curl http://localhost:8002/v1/models   # Should show Qwen3.5-9B
```

### VRAM budget (both models simultaneously)

| Model | Role | Precision | VRAM | Port |
| --- | --- | --- | --- | --- |
| SocratTeachLLM 9.4B | Teacher | BF16 | ~19GB | 8001 |
| Qwen3.5-9B | Consultant | BF16 | ~5GB | 8002 |
| **Total** | | | **~24GB / 32GB** | |

---

## Step 3 — Run the KELE System

Both models are served as OpenAI-compatible endpoints. The `.env` configures which ports to hit.

```bash
# .env file (create from template)
cp .env.example .env

# Contents:
CONSULTANT_API_KEY=not-needed
CONSULTANT_BASE_URL=http://localhost:8002/v1
CONSULTANT_MODEL_NAME=Qwen3.5-9B

TEACHER_API_KEY=not-needed
TEACHER_BASE_URL=http://localhost:8001/v1
TEACHER_MODEL_NAME=SocratTeachLLM
```

---

## Step 4 — Evaluation Pipeline with Metrics Export

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

## Step 5 — Phase 4: BERT Classifier Training

This trains the replacement consultant agent.

```bash
# Additional dependencies
pip install datasets scikit-learn

# Training (small model, fast even without wandb)
python src/project/consultant_classifier.py \
  --data references/KELE/SocratDataset.json \
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
# Activate venv (every new terminal)
source .venv/bin/activate

# Monitor GPU
watch -n 1 nvidia-smi

# Start both model servers
./scripts/serve_both.sh

# Or individually
./scripts/serve_socratteachllm.sh   # Teacher on port 8001
./scripts/serve_consultant.sh       # Consultant on port 8002

# Quick smoke test (3 dialogues)
python3 -m src.project.kele test --n 3 --output results/test

# Full baseline evaluation (680 dialogues)
./scripts/run_eval.sh

# Interactive session
python3 -m src.project.kele interactive
```

---

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `trust_remote_code` error | Add `--trust-remote-code` flag to vLLM (already in serve script) |
| OOM with both models | Reduce `--gpu-memory-utilization` in serve scripts, or reduce `--max-model-len` to 2048 |
| vLLM won't load ChatGLM | May need `--tokenizer-mode slow` for custom tokenizers |
| Slow generation | Check `--gpu-memory-utilization` isn't too low, confirm no CPU offload |
| Chinese output from SocratTeachLLM | Expected — model was trained on Chinese data, dataset is Chinese |
| NVML version mismatch | Reboot after driver update; PyTorch CUDA still works without NVML |
| Qwen3.5 text-only load error | Need vLLM >= 0.17 (we have 0.19.0, should be fine) |
