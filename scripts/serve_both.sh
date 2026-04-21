#!/usr/bin/env bash
# Serve both models simultaneously for the KELE pipeline
# SocratTeachLLM (teacher) on port 8001 — ~19GB VRAM
# Qwen3.5-9B (consultant) on port 8002 — ~5GB VRAM (Q4)
# Total: ~24GB of 32GB
#
# Usage: ./scripts/serve_both.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TEACHER_PORT="${TEACHER_PORT:-8001}"
CONSULTANT_PORT="${CONSULTANT_PORT:-8002}"

# ── Timestamped run directory ─────────────────────────────────────────────────
# All logs for this run land in one place: logs/YYYY-MM-DDTHH-MM-SS/
# Callers can override log paths via env vars before invoking this script.
RUN_DIR="${RUN_DIR:-logs/$(date -u +%Y-%m-%dT%H-%M-%S)}"
mkdir -p "$RUN_DIR"
TEACHER_LOG_FILE="${TEACHER_LOG_FILE:-$RUN_DIR/vllm_teacher.log}"
CONSULTANT_LOG_FILE="${CONSULTANT_LOG_FILE:-$RUN_DIR/vllm_consultant.log}"
export TEACHER_LOG_FILE CONSULTANT_LOG_FILE

echo "=== Starting KELE Model Servers ==="
echo "Run dir: $RUN_DIR"
echo "GPU status before:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
echo "---"

# Serialize startup: vLLM memory profiling races if both servers initialize
# at the same time (the teacher's KV cache budget goes negative if the
# consultant allocates weights mid-profile). Start teacher, wait for it to be
# fully ready, THEN start consultant.

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

echo "Starting SocratTeachLLM (teacher) on port $TEACHER_PORT..."
./scripts/serve_socratteachllm.sh &
TEACHER_PID=$!

wait_for_port "$TEACHER_PORT" || { echo "Teacher failed to start — see $TEACHER_LOG_FILE"; exit 1; }

echo ""
echo "Starting consultant on port $CONSULTANT_PORT..."
./scripts/serve_consultant.sh &
CONSULTANT_PID=$!

wait_for_port "$CONSULTANT_PORT" || { echo "Consultant failed to start — see $CONSULTANT_LOG_FILE"; exit 1; }

echo ""
echo "---"
echo "Teacher PID:     $TEACHER_PID (port $TEACHER_PORT)"
echo "Consultant PID:  $CONSULTANT_PID (port $CONSULTANT_PORT)"
echo "To stop both:    kill $TEACHER_PID $CONSULTANT_PID"

echo ""
echo "=== Both servers running ==="
echo "Test teacher:     curl http://localhost:$TEACHER_PORT/v1/models"
echo "Test consultant:  curl http://localhost:$CONSULTANT_PORT/v1/models"
echo "Run evaluation:   ./scripts/run_eval.sh"

wait
