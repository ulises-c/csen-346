#!/usr/bin/env bash
# Serve the teacher model locally for single-machine eval runs.
# No API key required — auth is disabled when TEACHER_SERVER_API_KEY is unset.
#
# Usage: ./scripts/serve_teacher_local.sh
# Then in a second terminal: ./scripts/run_eval.sh <experiment> --limit N

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="$(dirname "$0")/../configs/local-teacher.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=../configs/local-teacher.env
    source "$ENV_FILE"
    set +a
fi

HOST="${TEACHER_HOST:-0.0.0.0}"
PORT="${TEACHER_PORT:-8001}"

echo "=== Teacher (local) ==="
echo "Host: $HOST  Port: $PORT"
echo "Auth: disabled (no TEACHER_SERVER_API_KEY)"
echo "Test: curl http://localhost:$PORT/v1/models"
echo "---"

export TEACHER_HOST="$HOST"
export TEACHER_PORT="$PORT"

VENV_BIN=$(poetry env info --path 2>/dev/null)/bin
if [[ ! -x "$VENV_BIN/serve-teacher" ]]; then
    echo "ERROR: 'serve-teacher' entry point not found."
    echo "Run: poetry run pip install -e . --no-deps"
    exit 1
fi

exec poetry run serve-teacher
