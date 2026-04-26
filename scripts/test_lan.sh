#!/usr/bin/env bash
# Quick smoke test for the LAN consultant (Ollama on Mac Mini).
# Loads the experiment config and sends a minimal chat request.
#
# Usage:
#   ./scripts/test_lan.sh                    # uses local-mac-m4
#   ./scripts/test_lan.sh dual-local         # uses a different config

set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENT="${1:-local-mac-m4}"
ENV_FILE="configs/$EXPERIMENT.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: config not found: $ENV_FILE"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

URL="${CONSULTANT_BASE_URL}/chat/completions"
MODEL="${CONSULTANT_MODEL_NAME}"

echo "=== LAN consultant test ==="
echo "Config:  $EXPERIMENT"
echo "URL:     $URL"
echo "Model:   $MODEL"
echo ""

RAND=$RANDOM
PAYLOAD=$(printf '{"model":"%s","messages":[{"role":"user","content":"What is 2+2? Answer only in JSON like: {\\"answer\\": 4, \\"id\\": %d}"}],"max_tokens":250}' "$MODEL" "$RAND")

echo "Payload: $PAYLOAD"
echo ""
echo "Sending request..."
time curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${CONSULTANT_API_KEY:-ollama}" \
    -d "$PAYLOAD"
