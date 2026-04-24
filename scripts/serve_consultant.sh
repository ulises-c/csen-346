#!/usr/bin/env bash
# Serve the consultant model via vLLM.
# Usage: ./scripts/serve_consultant.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${CONSULTANT_LOCAL_MODEL_NAME:-Qwen3.5-9B}"
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
# NVIDIA: use compute capability (CC) to select dtype and CUDA graph support.
#   CC < 80  → V100/T4/RTX 20xx: no BF16, CUDA graphs unreliable → float16 + --enforce-eager
#   CC ≥ 80  → A100/L40S/RTX 30xx+: full BF16 + CUDA graphs
# AMD/ROCm: bfloat16 is supported on all modern RDNA2+ / CDNA cards.
#   --enforce-eager is NOT needed (ROCm doesn't use CUDA graphs).
GPU_VENDOR="unknown"
GPU_CC=0

if command -v nvidia-smi &>/dev/null; then
    _CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d '.' | tr -d ' ') || true
    if [[ -n "$_CC" ]]; then
        GPU_VENDOR="nvidia"
        GPU_CC="$_CC"
    fi
fi

if [[ "$GPU_VENDOR" == "unknown" ]] && command -v rocm-smi &>/dev/null; then
    GPU_VENDOR="amd"
fi

echo "GPU vendor: $GPU_VENDOR  (CC: ${GPU_CC:0:1}.${GPU_CC:1:-0})"

# dtype: env var override wins; otherwise auto-select by vendor/CC
if [[ -n "${CONSULTANT_DTYPE:-}" ]]; then
    DTYPE="$CONSULTANT_DTYPE"
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    DTYPE="bfloat16"
elif [[ "$GPU_CC" -ge 80 ]]; then
    DTYPE="bfloat16"
else
    DTYPE="float16"
fi

# --enforce-eager: NVIDIA CC < 8.0 only (AMD/ROCm doesn't benefit from this flag)
EXTRA_ARGS=()
if [[ "$GPU_VENDOR" == "nvidia" && "$GPU_CC" -lt 80 ]]; then
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
