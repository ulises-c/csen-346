#!/usr/bin/env bash
# Serve the consultant model via vLLM.
# Usage: ./scripts/serve_consultant.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${CONSULTANT_LOCAL_MODEL_NAME:-Qwen3.5-4B}"
MODEL_PATH="${CONSULTANT_MODEL_PATH:-${HF_HOME:-$HOME/hf_models}/$MODEL_NAME}"
HOST="${CONSULTANT_HOST:-0.0.0.0}"
PORT="${CONSULTANT_PORT:-8002}"
MAX_MODEL_LEN="${CONSULTANT_MAX_MODEL_LEN:-6144}"
GPU_MEMORY_UTILIZATION="${CONSULTANT_GPU_MEMORY_UTILIZATION:-0.30}"
LOG_FILE="${CONSULTANT_LOG_FILE:-logs/vllm_consultant.log}"

mkdir -p logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download it first:"
    echo "  hf download Qwen/$MODEL_NAME --local-dir $MODEL_PATH"
    exit 1
fi

# ── GPU architecture detection ────────────────────────────────────────────────
# Detect compute capability of the first visible GPU (e.g. "7.0" → 70, "8.9" → 89).
# CC < 80  → V100/T4-class: no BF16, CUDA graphs unreliable → float16 + --enforce-eager
# CC ≥ 80  → A100/L40S/RTX 5090-class: full BF16 + CUDA graphs
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | head -1 | tr -d '.' | tr -d ' ')
GPU_CC=${GPU_CC:-0}
echo "GPU compute capability: ${GPU_CC:0:1}.${GPU_CC:1}"

# dtype: env var override wins; otherwise auto-select by CC
if [[ -n "${CONSULTANT_DTYPE:-}" ]]; then
    DTYPE="$CONSULTANT_DTYPE"
elif [[ "$GPU_CC" -ge 80 ]]; then
    DTYPE="bfloat16"
else
    DTYPE="float16"
fi

# --enforce-eager: auto-enable on CC < 8.0 (skips CUDA graph compilation,
# avoids V100 graph errors and speeds up cold start on low-CC hardware)
EXTRA_ARGS=()
if [[ "$GPU_CC" -lt 80 ]]; then
    EXTRA_ARGS+=(--enforce-eager)
fi

echo "Serving $MODEL_NAME (consultant) on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Host: $HOST"
echo "DTYPE: $DTYPE"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
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
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
