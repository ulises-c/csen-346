#!/usr/bin/env bash
# Serve Qwen3.6-27B UD-Q5_K_XL (clean variant, NOT the HauhauCS uncensored fine-tune)
# as an OpenAI-compatible API server on the RTX 5090.
#
# 416K context: ~19 GB model + ~7.3 GB KV (Q4_0) = ~26.3 GB. ~6 GB headroom.
# 6 parallel slots, unified KV — both the KELE teacher and consultant hit this
# same server (configs/qwen27b-local.env).
#
# Endpoint: http://localhost:8080/v1/chat/completions
#
# Override the weights dir with:
#   QWEN27B_WEIGHTS_DIR=/some/other/path ./scripts/serve_qwen27b_q5.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

QWEN27B_WEIGHTS_DIR="${QWEN27B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${QWEN27B_WEIGHT_FILE:-Qwen3.6-27B-UD-Q5_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_qwen27b.sh" \
  -m "$QWEN27B_WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Qwen 27B Q5" \
  -c 425984 \
  "$@"
