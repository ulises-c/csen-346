#!/usr/bin/env bash
# Start the Ollama consultant server and verify it's reachable.
# Used when running the KELE eval from a host PC against a Mac Mini consultant.
#
# Run this on the Mac Mini before starting the eval on the host PC.
# If Ollama.app is already running with OLLAMA_HOST=0.0.0.0, this script just
# verifies connectivity and prints the endpoint — no extra process needed.
#
# Usage: ./scripts/serve_consultant_ollama.sh

set -euo pipefail

HOST="${CONSULTANT_HOST:-0.0.0.0}"
PORT="${CONSULTANT_PORT:-11434}"
MODEL="${CONSULTANT_MODEL_NAME:-qwen3.5:9b}"
LOG_FILE="${CONSULTANT_LOG_FILE:-logs/ollama_consultant.log}"

mkdir -p logs

echo "=== Ollama Consultant (Mac Mini) ==="
echo "Model:  $MODEL"
echo "Host:   $HOST"
echo "Port:   $PORT"
echo "Log:    $LOG_FILE"
echo ""

# Check if Ollama is already running and listening on the expected port.
if curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
    echo "Ollama is already running on port $PORT."
    LISTEN=$(lsof -iTCP:$PORT -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1 {print $9}' | head -1)
    echo "Binding: $LISTEN"

    # Verify the model is available.
    if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; then
        echo "Model '$MODEL' confirmed available."
    else
        echo "Model '$MODEL' not found — pulling..."
        ollama pull "$MODEL"
    fi
else
    echo "Ollama not running — starting with OLLAMA_HOST=$HOST..."
    OLLAMA_HOST="$HOST" ollama serve > "$LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    echo "PID: $OLLAMA_PID  Log: $LOG_FILE"

    echo -n "Waiting for Ollama to be ready..."
    for i in $(seq 1 30); do
        if curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
            echo " ready! (~${i}s)"
            break
        fi
        sleep 1
    done

    if ! curl -s "http://localhost:$PORT/api/version" > /dev/null 2>&1; then
        echo "ERROR: Ollama failed to start. Check $LOG_FILE"
        exit 1
    fi

    # Warm up: a single short generation so the first real eval call is fast.
    echo "Warming up model (loading weights into memory)..."
    curl -s http://localhost:$PORT/api/generate \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"hi\", \"stream\": false}" \
        > /dev/null
    echo "Model loaded."
fi

MAC_HOSTNAME=$(scutil --get LocalHostName 2>/dev/null || echo "")
MAC_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<mac-ip>")
MAC_ADDR="${MAC_HOSTNAME:+${MAC_HOSTNAME}.local}"
MAC_ADDR="${MAC_ADDR:-$MAC_IP}"

echo ""
echo "=== Consultant ready ==="
echo "OpenAI-compatible endpoint: http://${MAC_ADDR}:${PORT}/v1"
echo ""
echo "On the host PC, set:"
echo "  export CONSULTANT_BASE_URL=http://${MAC_ADDR}:${PORT}/v1"
echo "  export CONSULTANT_MODEL_NAME=${MODEL}"
echo "  export CONSULTANT_API_KEY=ollama"
echo ""
echo "Test from host PC:"
echo "  curl http://${MAC_ADDR}:${PORT}/v1/models"
