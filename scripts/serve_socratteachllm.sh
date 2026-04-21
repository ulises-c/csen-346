#!/usr/bin/env bash
# Serve SocratTeachLLM (teacher) via vLLM.
# Usage: ./scripts/serve_socratteachllm.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${TEACHER_MODEL_NAME:-SocratTeachLLM}"
MODEL_PATH="${TEACHER_MODEL_PATH:-${HF_HOME:-$HOME/hf_models}/SocratTeachLLM}"
HOST="${TEACHER_HOST:-0.0.0.0}"
PORT="${TEACHER_PORT:-8001}"
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

# ── GPU architecture detection ────────────────────────────────────────────────
# Detect compute capability of the first visible GPU (e.g. "7.0" → 70, "8.9" → 89).
# CC < 80  → V100/T4-class: no BF16, CUDA graphs unreliable → float16 + --enforce-eager
# CC ≥ 80  → A100/L40S/RTX 5090-class: full BF16 + CUDA graphs
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | head -1 | tr -d '.' | tr -d ' ')
GPU_CC=${GPU_CC:-0}
echo "GPU compute capability: ${GPU_CC:0:1}.${GPU_CC:1}"

# dtype: env var override wins; otherwise auto-select by CC
if [[ -n "${TEACHER_DTYPE:-}" ]]; then
    DTYPE="$TEACHER_DTYPE"
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

echo "Serving $MODEL_NAME (teacher) on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Host: $HOST"
echo "DTYPE: $DTYPE"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
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
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
