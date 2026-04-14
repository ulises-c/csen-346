#!/usr/bin/env bash
# Serve Qwen3.5-9B (consultant) via vLLM on port 8002
# Usage: ./scripts/serve_consultant.sh

set -euo pipefail

MODEL_PATH="${HF_HOME:-$HOME/hf_models}/Qwen3.5-9B"
PORT=8002
LOG_FILE="logs/vllm_consultant.log"

mkdir -p logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download it first:"
    echo "  huggingface-cli download Qwen/Qwen3.5-9B --local-dir $MODEL_PATH"
    exit 1
fi

echo "Serving Qwen3.5-9B (consultant) on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Log: $LOG_FILE"
echo "Test: curl http://localhost:$PORT/v1/models"
echo "---"

exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.30 \
    2>&1 | tee "$LOG_FILE"
