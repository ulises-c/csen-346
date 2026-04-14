#!/usr/bin/env bash
# Serve Gemma-4-31B-IT-NVFP4 (teacher, Run 2) via vLLM on port 8001
# Usage: ./scripts/serve_gemma4.sh
#
# VRAM budget (with consultant):
#   Gemma 4 31B NVFP4 ~17GB + Qwen3.5-9B ~5GB = ~22GB / 32GB

set -euo pipefail

MODEL_PATH="${HF_HOME:-$HOME/hf_models}/Gemma-4-31B-IT-NVFP4"
PORT=8001
LOG_FILE="logs/vllm_gemma4.log"

mkdir -p logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download it first:"
    echo "  huggingface-cli download nvidia/Gemma-4-31B-IT-NVFP4 --local-dir $MODEL_PATH"
    exit 1
fi

echo "Serving Gemma-4-31B-IT-NVFP4 (teacher, Run 2) on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Log: $LOG_FILE"
echo "Test: curl http://localhost:$PORT/v1/models"
echo "---"

exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --quantization modelopt \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.55 \
    2>&1 | tee "$LOG_FILE"
