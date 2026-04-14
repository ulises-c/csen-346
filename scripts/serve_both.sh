#!/usr/bin/env bash
# Serve both models simultaneously for the KELE pipeline
# SocratTeachLLM (teacher) on port 8001 — ~19GB VRAM
# Qwen3.5-9B (consultant) on port 8002 — ~5GB VRAM (Q4)
# Total: ~24GB of 32GB
#
# Usage: ./scripts/serve_both.sh

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

echo "=== Starting KELE Model Servers ==="
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

echo "Starting SocratTeachLLM (teacher) on port 8001..."
./scripts/serve_socratteachllm.sh &
TEACHER_PID=$!

wait_for_port 8001 || { echo "Teacher failed to start — see logs/vllm_socrat.log"; exit 1; }

echo ""
echo "Starting Qwen3.5-2B (consultant) on port 8002..."
./scripts/serve_consultant.sh &
CONSULTANT_PID=$!

wait_for_port 8002 || { echo "Consultant failed to start — see logs/vllm_consultant.log"; exit 1; }

echo ""
echo "---"
echo "Teacher PID:     $TEACHER_PID (port 8001)"
echo "Consultant PID:  $CONSULTANT_PID (port 8002)"
echo "To stop both:    kill $TEACHER_PID $CONSULTANT_PID"

echo ""
echo "=== Both servers running ==="
echo "Test teacher:     curl http://localhost:8001/v1/models"
echo "Test consultant:  curl http://localhost:8002/v1/models"
echo "Run evaluation:   ./scripts/run_eval.sh"

wait
