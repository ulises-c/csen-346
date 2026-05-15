#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B (MoE) UD-Q4_K_M with reasoning ENABLED at the server.
#
# Companion to serve_qwen35b_a3b.sh — identical except --reasoning off is NOT
# passed, so Qwen3's chat-template thinking mechanism is honored. Use this for
# matched-n gradient experiments where we explicitly want to compare to the
# tournament's no-think result.
#
# The eval orchestrator's alias verification will bail if the wrong one is up.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

QWEN35B_WEIGHTS_DIR="${QWEN35B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${QWEN35B_WEIGHT_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"

exec "$SCRIPT_DIR/serve_qwen27b.sh" \
  -m "$QWEN35B_WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Qwen 35B A3B" \
  -c 524288 \
  "$@"
