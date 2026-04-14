#!/usr/bin/env bash
# Serve SocratTeachLLM via vLLM on port 8001
# Usage: ./scripts/serve_socratteachllm.sh

set -euo pipefail

MODEL_PATH="${HF_HOME:-$HOME/hf_models}/SocratTeachLLM"
PORT=8001
LOG_FILE="logs/vllm_socrat.log"

mkdir -p logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download it first:"
    echo "  huggingface-cli download yuanpan/SocratTeachLLM --local-dir $MODEL_PATH"
    exit 1
fi

echo "Serving SocratTeachLLM on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Log: $LOG_FILE"
echo "Test: curl http://localhost:$PORT/v1/models"
echo "---"

exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    2>&1 | tee "$LOG_FILE"
