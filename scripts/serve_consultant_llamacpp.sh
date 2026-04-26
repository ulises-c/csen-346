#!/usr/bin/env bash
# Start the llama.cpp consultant server and verify it's reachable.
# Used when running the KELE eval from a host PC against a Mac Mini consultant.
#
# Run this on the Mac Mini before starting the eval on the host PC.
# If llama-server is already running on the expected port, this script just
# verifies connectivity and prints the endpoint — no extra process needed.
#
# Usage: ./scripts/serve_consultant_llamacpp.sh

set -euo pipefail

HOST="${CONSULTANT_HOST:-0.0.0.0}"
PORT="${CONSULTANT_PORT:-8080}"
MODEL_PATH="${CONSULTANT_MODEL_PATH:?Set CONSULTANT_MODEL_PATH to your .gguf file (e.g. source configs/local-mac-m4.env)}"
LOG_FILE="${CONSULTANT_LOG_FILE:-logs/llamacpp_consultant.log}"
CTX="${CONSULTANT_NUM_CTX:-16384}"
GPU_LAYERS="${CONSULTANT_GPU_LAYERS:-99}"  # M4 Mac Mini: all layers to Metal GPU

mkdir -p logs

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    echo "  Set CONSULTANT_MODEL_PATH to the absolute path of your .gguf file."
    exit 1
fi

echo "=== llama.cpp Consultant (Mac Mini) ==="
echo "Model:      $MODEL_PATH"
echo "Host:       $HOST"
echo "Port:       $PORT"
echo "CTX:        $CTX"
echo "GPU layers: $GPU_LAYERS  (-ngl)"
echo "Log:        $LOG_FILE"
echo ""

# Check if llama-server is already running on the expected port.
if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "llama-server is already running on port $PORT."
    LISTEN=$(lsof -iTCP:$PORT -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1 {print $9}' | head -1)
    echo "Binding: $LISTEN"

    LLAMA_PID=$(lsof -iTCP:$PORT -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1 {print $2}' | head -1)
    if [[ -n "$LLAMA_PID" ]] && command -v caffeinate &>/dev/null; then
        caffeinate -s -w "$LLAMA_PID" &
        echo "Sleep inhibited (caffeinate -s -w $LLAMA_PID)."
    fi
else
    echo "llama-server not running — starting..."

    llama-server \
        --model "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --ctx-size "$CTX" \
        --n-gpu-layers "$GPU_LAYERS" \
        > "$LOG_FILE" 2>&1 &
    LLAMA_PID=$!
    echo "PID: $LLAMA_PID  Log: $LOG_FILE"

    if command -v caffeinate &>/dev/null; then
        caffeinate -s -w "$LLAMA_PID" &
        echo "Sleep inhibited (caffeinate -s -w $LLAMA_PID)."
    fi

    echo -n "Waiting for llama-server to be ready..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo " ready! (~${i}s)"
            break
        fi
        sleep 1
    done

    if ! curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "ERROR: llama-server failed to start. Check $LOG_FILE"
        exit 1
    fi

    # Warm up: load weights into GPU memory so the first eval call is fast.
    echo "Warming up model (loading weights into Metal GPU memory)..."
    curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"consultant","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
        > /dev/null
    echo "Model loaded."
fi

MAC_HOSTNAME=$(scutil --get LocalHostName 2>/dev/null || echo "")
MAC_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<mac-ip>")
MAC_ADDR="${MAC_HOSTNAME:+${MAC_HOSTNAME}.local}"
MAC_ADDR="${MAC_ADDR:-$MAC_IP}"

MODEL_NAME="${CONSULTANT_MODEL_NAME:-$(basename "$MODEL_PATH" .gguf)}"

echo ""
echo "=== Consultant ready ==="
echo "OpenAI-compatible endpoint: http://${MAC_ADDR}:${PORT}/v1"
echo ""
echo "On the host PC, set:"
echo "  export CONSULTANT_BASE_URL=http://${MAC_ADDR}:${PORT}/v1"
echo "  export CONSULTANT_MODEL_NAME=${MODEL_NAME}"
echo "  export CONSULTANT_API_KEY=no-key"
echo ""
echo "Test from host PC:"
echo "  curl http://${MAC_ADDR}:${PORT}/v1/models"
