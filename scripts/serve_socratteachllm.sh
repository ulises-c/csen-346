#!/usr/bin/env bash
# Serve SocratTeachLLM (teacher) via vLLM.
# Usage: ./scripts/serve_socratteachllm.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${TEACHER_MODEL_NAME:-SocratTeachLLM}"
MODEL_PATH="${TEACHER_MODEL_PATH:-${HF_HOME:-$HOME/hf_models}/SocratTeachLLM}"
HOST="${TEACHER_HOST:-0.0.0.0}"
PORT="${TEACHER_PORT:-8001}"
DTYPE="${TEACHER_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.60}"
LOG_FILE="${TEACHER_LOG_FILE:-logs/vllm_socrat.log}"

mkdir -p logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download it first:"
    echo "  hf download ulises-c/SocratTeachLLM --local-dir $MODEL_PATH"
    exit 1
fi

echo "Serving $MODEL_NAME (teacher) on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Host: $HOST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Log: $LOG_FILE"
echo "Test: curl http://localhost:$PORT/v1/models"
echo "---"

exec poetry run vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    2>&1 | tee "$LOG_FILE"
