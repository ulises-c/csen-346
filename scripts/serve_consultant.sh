#!/usr/bin/env bash
# Serve the consultant model via vLLM.
# Usage: ./scripts/serve_consultant.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${CONSULTANT_LOCAL_MODEL_NAME:-Qwen3.5-4B}"
MODEL_PATH="${CONSULTANT_MODEL_PATH:-${HF_HOME:-$HOME/hf_models}/$MODEL_NAME}"
HOST="${CONSULTANT_HOST:-0.0.0.0}"
PORT="${CONSULTANT_PORT:-8002}"
DTYPE="${CONSULTANT_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${CONSULTANT_MAX_MODEL_LEN:-6144}"
GPU_MEMORY_UTILIZATION="${CONSULTANT_GPU_MEMORY_UTILIZATION:-0.30}"
LOG_FILE="${CONSULTANT_LOG_FILE:-logs/vllm_consultant.log}"

mkdir -p logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download it first:"
    echo "  huggingface-cli download Qwen/$MODEL_NAME --local-dir $MODEL_PATH"
    exit 1
fi

echo "Serving $MODEL_NAME (consultant) on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Host: $HOST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Log: $LOG_FILE"
echo "Test: curl http://localhost:$PORT/v1/models"
echo "---"

export PYTORCH_ALLOC_CONF=expandable_segments:True

exec poetry run vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    2>&1 | tee "$LOG_FILE"
