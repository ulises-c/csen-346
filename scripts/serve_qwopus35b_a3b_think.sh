#!/usr/bin/env bash
# Serve Qwopus3.6-35B-A3B-v1 (Q4_K_M) with Qwen3 thinking ENABLED (no --reasoning off).
# Companion to serve_qwopus35b_a3b.sh; identical except thinking is left on so
# the gradient measurement (think vs. no-think) for this MoE variant is meaningful.
#
# Local weights path overridable via QWOPUS35B_WEIGHT_FILE; default points to
# the same ~/Documents/models/weights/ tree the 5090 rig uses for all GGUFs.
#
# Boot:  bash scripts/serve_qwopus35b_a3b_think.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHT_FILE="${QWOPUS35B_WEIGHT_FILE:-$HOME/Documents/models/weights/Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf}"

exec "$SCRIPT_DIR/serve_qwen27b.sh" \
  -m "$WEIGHT_FILE" \
  -a "Qwopus 35B A3B" \
  -c 32768 \
  "$@"
