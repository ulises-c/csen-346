#!/usr/bin/env bash
# Run evaluation for a specific experiment
#
# Usage:
#   ./scripts/run_eval.sh baseline          # SocratTeachLLM teacher (default)
#   ./scripts/run_eval.sh gemma4            # Gemma 4 31B teacher
#   ./scripts/run_eval.sh baseline --limit 10   # Quick partial run

set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENT="${1:-baseline}"
shift 2>/dev/null || true  # shift past experiment name, remaining args pass through

echo "=== KELE Evaluation: $EXPERIMENT ==="
echo "Start time: $(date)"
echo "Config: configs/$EXPERIMENT.env"
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null \
    || rocm-smi --showproductname --showmeminfo vram --showtemp 2>/dev/null \
    || echo "(nvidia-smi/rocm-smi unavailable)"
echo "---"

INHIBIT=""
if command -v systemd-inhibit &>/dev/null; then
    INHIBIT="systemd-inhibit --what=sleep:idle --who=run_eval.sh --why=KELE-eval"
fi

$INHIBIT poetry run python -m src.project.kele --experiment "$EXPERIMENT" evaluate \
    --output "results/$EXPERIMENT" \
    "$@"

echo "---"
echo "End time: $(date)"
echo "Results: results/$EXPERIMENT/"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null \
    || rocm-smi --showproductname --showmeminfo vram --showtemp 2>/dev/null \
    || echo "(nvidia-smi/rocm-smi unavailable)"
