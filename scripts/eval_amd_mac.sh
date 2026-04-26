#!/usr/bin/env bash
# Run KELE eval: AMD Linux teacher + Mac Mini llama.cpp consultant.
#
# Usage:
#   ./scripts/eval_amd_mac.sh           # full run
#   ./scripts/eval_amd_mac.sh --limit 5 # quick smoke test

set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT_ARGS=()
if [[ "${1:-}" == "--limit" && -n "${2:-}" ]]; then
    LIMIT_ARGS=(--limit "$2")
elif [[ "${1:-}" =~ ^--limit=([0-9]+)$ ]]; then
    LIMIT_ARGS=(--limit "${BASH_REMATCH[1]}")
fi

# ── Load teacher server config ────────────────────────────────────────────────
set -a
# shellcheck source=../configs/teachers/socrat-r9700.env
source configs/teachers/socrat-r9700.env
set +a

TEACHER_PORT="${TEACHER_PORT:-8001}"
CONSULTANT_URL="${CONSULTANT_BASE_URL:-http://Ulisess-Mac-mini.local:8080/v1}"

# ── Verify Mac Mini consultant is reachable ───────────────────────────────────
echo "Checking Mac Mini consultant at $CONSULTANT_URL ..."
if ! curl -sf "${CONSULTANT_URL}/models" > /dev/null; then
    echo "ERROR: Mac Mini consultant unreachable. Run serve_consultant_llamacpp.sh on the Mac Mini first."
    exit 1
fi
echo "Consultant OK."
echo ""

# ── Start teacher in background ───────────────────────────────────────────────
echo "Starting teacher server on port $TEACHER_PORT..."
./scripts/serve_teacher_local.sh &
TEACHER_PID=$!
trap 'echo "Stopping teacher (PID $TEACHER_PID)..."; kill "$TEACHER_PID" 2>/dev/null; wait "$TEACHER_PID" 2>/dev/null || true' EXIT

echo -n "Waiting for teacher to be ready..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$TEACHER_PORT/v1/models" > /dev/null 2>&1; then
        echo " ready! (~${i}s)"
        break
    fi
    sleep 1
done

if ! curl -sf "http://localhost:$TEACHER_PORT/v1/models" > /dev/null 2>&1; then
    echo "ERROR: Teacher failed to start after 120s."
    exit 1
fi

# ── Run eval ──────────────────────────────────────────────────────────────────
echo ""
./scripts/run_eval.sh local-mac-m4 "${LIMIT_ARGS[@]}"
