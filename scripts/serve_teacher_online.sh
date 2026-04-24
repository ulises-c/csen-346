#!/usr/bin/env bash
# Serve the local teacher model safely for an external tunnel / reverse proxy.
# Binds to localhost by default and requires TEACHER_SERVER_API_KEY.
#
# Usage:
#   TEACHER_SERVER_API_KEY=change-me ./scripts/serve_teacher_online.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${TEACHER_SERVER_API_KEY:-}" ]; then
    echo "ERROR: TEACHER_SERVER_API_KEY must be set for online serving."
    echo "Example:"
    echo "  TEACHER_SERVER_API_KEY=change-me ./scripts/serve_teacher_online.sh"
    exit 1
fi

export TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
export TEACHER_PORT="${TEACHER_PORT:-8001}"

echo "Serving teacher model on ${TEACHER_HOST}:${TEACHER_PORT}"
echo "Auth required via Authorization: Bearer <key> or x-api-key"
echo "Health check: http://${TEACHER_HOST}:${TEACHER_PORT}/healthz"
echo "Model list:   http://${TEACHER_HOST}:${TEACHER_PORT}/v1/models"

VENV_BIN=$(poetry env info --path 2>/dev/null)/bin
if [[ ! -x "$VENV_BIN/serve-teacher" ]]; then
    echo "ERROR: 'serve-teacher' entry point not found."
    echo "Run: poetry run pip install -e . --no-deps"
    exit 1
fi

exec poetry run serve-teacher
