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

# Start teacher (SocratTeachLLM) in background
echo "Starting SocratTeachLLM (teacher) on port 8001..."
./scripts/serve_socratteachllm.sh &
TEACHER_PID=$!

# Wait a moment then start consultant
sleep 5

echo "Starting Qwen3.5-9B (consultant) on port 8002..."
./scripts/serve_consultant.sh &
CONSULTANT_PID=$!

echo "---"
echo "Teacher PID:     $TEACHER_PID (port 8001)"
echo "Consultant PID:  $CONSULTANT_PID (port 8002)"
echo "To stop both:    kill $TEACHER_PID $CONSULTANT_PID"
echo ""
echo "Waiting for models to load... (check logs/ for details)"

# Wait for both servers to be ready
for port in 8001 8002; do
    echo -n "Waiting for port $port..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo " ready!"
            break
        fi
        sleep 2
    done
done

echo ""
echo "=== Both servers running ==="
echo "Test teacher:     curl http://localhost:8001/v1/models"
echo "Test consultant:  curl http://localhost:8002/v1/models"
echo "Run evaluation:   ./scripts/run_eval.sh"

wait
