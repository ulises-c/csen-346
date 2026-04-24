#!/usr/bin/env bash
# One-time setup: configure Ollama on the Mac Mini to accept remote connections.
#
# Run this once on the Mac Mini. After running, the host PC can reach
# Ollama at http://<mac-ip>:11434 (OpenAI-compatible API on /v1).
#
# Usage: bash scripts/mac_mini_setup.sh

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

echo -e "${BOLD}Mac Mini Ollama Remote Access Setup${NC}"
echo "======================================"
date
echo ""

# ── 1. Verify Ollama is installed ─────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    die "Ollama not found in PATH. Install it from https://ollama.com then re-run."
fi
info "Ollama found: $(ollama --version)"

# ── 2. Verify the consultant model is pulled ───────────────────────────────────
MODEL="${CONSULTANT_MODEL_NAME:-}"
if [[ -z "$MODEL" ]]; then
    die "CONSULTANT_MODEL_NAME is not set. Source your experiment .env first, e.g.:\n  set -a && source configs/R9700_Mac-M4.env && set +a"
fi
if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
    warn "$MODEL not found — pulling now..."
    ollama pull "$MODEL"
else
    info "$MODEL already present."
fi

# ── 3. Set OLLAMA_HOST and OLLAMA_NUM_CTX ─────────────────────────────────────
# Ollama defaults to 127.0.0.1:11434, which rejects remote connections.
# Setting OLLAMA_HOST=0.0.0.0 makes it bind to all interfaces.
# OLLAMA_NUM_CTX sets the default context window (default is ~2048; we need
# 16384 so multi-turn Socratic dialogues fit in a single call).
#
# For the Ollama.app (Electron), env vars must be set before the helper process
# launches. launchctl setenv injects them into the macOS GUI session environment
# for all future launches.
info "Setting OLLAMA_HOST=0.0.0.0 in launchctl environment..."
launchctl setenv OLLAMA_HOST 0.0.0.0
info "Setting OLLAMA_NUM_CTX=16384 in launchctl environment..."
launchctl setenv OLLAMA_NUM_CTX 16384

# ── 4. Restart Ollama to pick up the new env var ──────────────────────────────
# The running instance still has the old binding (127.0.0.1). Kill it so the
# launchd re-spawn picks up OLLAMA_HOST from the environment we just set.
info "Restarting Ollama service..."
if pgrep -x "ollama" > /dev/null 2>&1; then
    pkill -x "ollama" || true
    sleep 2
fi

# Re-launch via the app bundle if present, otherwise let launchd handle it.
if [[ -d "/Applications/Ollama.app" ]]; then
    open -a Ollama
    sleep 3
else
    warn "Ollama.app not found at /Applications/Ollama.app."
    warn "Start Ollama manually: OLLAMA_HOST=0.0.0.0 ollama serve"
fi

# ── 5. Verify Ollama is listening on 0.0.0.0 ─────────────────────────────────
info "Waiting for Ollama to come back up..."
for i in $(seq 1 15); do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

LISTEN=$(lsof -iTCP:11434 -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1 {print $9}' | head -1)
if [[ "$LISTEN" == *"0.0.0.0"* ]] || [[ "$LISTEN" == *"*"* ]]; then
    info "Ollama is listening on all interfaces: $LISTEN"
else
    warn "Ollama may still be bound to localhost only: $LISTEN"
    warn "If the host PC can't connect, quit Ollama.app and run:"
    warn "  OLLAMA_HOST=0.0.0.0 ollama serve"
fi

# ── 6. Print this machine's address ──────────────────────────────────────────
MAC_HOSTNAME=$(scutil --get LocalHostName 2>/dev/null || echo "")
MAC_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "unknown")
MAC_ADDR="${MAC_HOSTNAME:+${MAC_HOSTNAME}.local}"
MAC_ADDR="${MAC_ADDR:-$MAC_IP}"
echo ""
echo -e "${BOLD}Mac Mini address: $MAC_ADDR${NC}"
echo "(IP fallback: $MAC_IP)"
echo ""
echo "From the host PC, set these before running the eval:"
echo ""
echo "  export CONSULTANT_BASE_URL=http://${MAC_ADDR}:11434/v1"
echo "  export CONSULTANT_MODEL_NAME=${MODEL}"
echo "  export CONSULTANT_API_KEY=ollama"
echo ""
echo "Then run the evaluation against the local consultant:"
echo ""
echo "  ./scripts/run_eval.sh local"
echo ""

# ── 7. macOS firewall reminder ────────────────────────────────────────────────
warn "If the host PC can't reach port 11434, check macOS firewall:"
warn "  System Settings → Network → Firewall → Options"
warn "  Make sure Ollama is not blocked, or temporarily disable the firewall."
echo ""
info "Setup complete. See scripts/MAC_MINI_SETUP.md for full documentation."
