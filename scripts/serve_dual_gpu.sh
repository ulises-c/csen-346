#!/usr/bin/env bash
# Serve both models on a dual-GPU system (e.g. 2× L40S 48GB).
# Teacher (SocratTeachLLM) is pinned to GPU 0; consultant to GPU 1.
# Each model gets an isolated 48GB card so GPU_MEMORY_UTILIZATION can be
# raised well above the shared-card defaults in serve_both.sh.
#
# Usage: ./scripts/serve_dual_gpu.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER_PORT="${TEACHER_PORT:-8001}"
CONSULTANT_PORT="${CONSULTANT_PORT:-8002}"

# ── Timestamped run directory ─────────────────────────────────────────────────
RUN_DIR="${RUN_DIR:-logs/$(date -u +%Y-%m-%dT%H-%M-%S)}"
mkdir -p "$RUN_DIR"
TEACHER_LOG_FILE="${TEACHER_LOG_FILE:-$RUN_DIR/vllm_teacher.log}"
CONSULTANT_LOG_FILE="${CONSULTANT_LOG_FILE:-$RUN_DIR/vllm_consultant.log}"
export TEACHER_LOG_FILE CONSULTANT_LOG_FILE

echo "=== Dual-GPU KELE Model Servers ==="
echo "Run dir: $RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
echo "---"

wait_for_port() {
    local port=$1
    echo -n "Waiting for port $port..."
    for i in $(seq 1 180); do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo " ready! (~$((i*2))s)"
            return 0
        fi
        sleep 2
    done
    echo " TIMEOUT after 360s"
    return 1
}

# GPU 0 → teacher (dedicated 48GB card: raise utilization to 0.85)
echo "Starting SocratTeachLLM (teacher) on GPU 0, port $TEACHER_PORT..."
CUDA_VISIBLE_DEVICES=0 \
  TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}" \
  TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-8192}" \
  ./scripts/serve_socratteachllm.sh &
TEACHER_PID=$!

wait_for_port "$TEACHER_PORT" || { echo "Teacher failed to start — see $TEACHER_LOG_FILE"; exit 1; }

echo ""
echo "Starting consultant on GPU 1, port $CONSULTANT_PORT..."
CUDA_VISIBLE_DEVICES=1 \
  CONSULTANT_GPU_MEMORY_UTILIZATION="${CONSULTANT_GPU_MEMORY_UTILIZATION:-0.85}" \
  CONSULTANT_MAX_MODEL_LEN="${CONSULTANT_MAX_MODEL_LEN:-8192}" \
  ./scripts/serve_consultant.sh &
CONSULTANT_PID=$!

wait_for_port "$CONSULTANT_PORT" || { echo "Consultant failed to start — see $CONSULTANT_LOG_FILE"; exit 1; }

echo ""
echo "---"
echo "Teacher PID:     $TEACHER_PID  (GPU 0, port $TEACHER_PORT)"
echo "Consultant PID:  $CONSULTANT_PID  (GPU 1, port $CONSULTANT_PORT)"
echo "To stop both:    kill $TEACHER_PID $CONSULTANT_PID"
echo ""
echo "=== Both servers running ==="
echo "Test teacher:     curl http://localhost:$TEACHER_PORT/v1/models"
echo "Test consultant:  curl http://localhost:$CONSULTANT_PORT/v1/models"
echo "Run evaluation:   ./scripts/run_eval.sh"

wait
