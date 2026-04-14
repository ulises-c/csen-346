# After Reboot — Launch Overnight Baseline Eval

```bash
# 1. Verify GPU is clean
nvidia-smi

# 2. Activate venv
cd ~/Documents/scu/CSEN-346/csen-346
source .venv/bin/activate

# 3. Start teacher model (SocratTeachLLM on port 8001) — run in tmux or separate terminal
vllm serve ~/hf_models/SocratTeachLLM \
  --host 0.0.0.0 --port 8001 --dtype bfloat16 \
  --trust-remote-code --max-model-len 4096 \
  --gpu-memory-utilization 0.55 2>&1 | tee logs/vllm_socrat.log

# 4. Start consultant model (Qwen3.5-2B on port 8002) — another tmux pane or terminal
vllm serve ~/hf_models/Qwen3.5-2B \
  --host 0.0.0.0 --port 8002 --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.30 2>&1 | tee logs/vllm_consultant.log

# 5. Wait for both servers to be ready, then test
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models

# 6. Smoke test (3 dialogues)
python3 -m src.project.kele -e baseline test --n 3 --output results/test

# 7. If smoke test passes — kick off full overnight eval (681 dialogues)
./scripts/run_eval.sh baseline
```
