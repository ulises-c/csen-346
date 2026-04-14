#!/usr/bin/env bash
# Serve Qwen3.5-4B (consultant) via vLLM on port 8002
# Usage: ./scripts/serve_consultant.sh

set -euo pipefail

MODEL_NAME="Qwen3.5-4B"
MODEL_PATH="${HF_HOME:-$HOME/hf_models}/$MODEL_NAME"
PORT=8002
LOG_FILE="logs/vllm_consultant.log"

mkdir -p logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download it first:"
    echo "  huggingface-cli download Qwen/$MODEL_NAME --local-dir $MODEL_PATH"
    exit 1
fi

echo "Serving $MODEL_NAME (consultant) on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Log: $LOG_FILE"
echo "Test: curl http://localhost:$PORT/v1/models"
echo "---"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Teacher occupies ~19 GB of 32 GB; 0.32 util ≈ 10 GB is the practical ceiling
# for the consultant. 4B weights (~8 GB bf16) + KV cache at 8192 ctx fit.
exec poetry run vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 6144 \
    --gpu-memory-utilization 0.30 \
    --enforce-eager \
    2>&1 | tee "$LOG_FILE"
