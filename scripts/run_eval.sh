#!/usr/bin/env bash
# Run the full baseline evaluation overnight
# Usage: ./scripts/run_eval.sh [--limit N] [--start-id N]

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== KELE Baseline Evaluation ==="
echo "Start time: $(date)"
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo "---"

python3 -m src.project.kele evaluate \
    --output results/baseline \
    "$@"

echo "---"
echo "End time: $(date)"
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader
