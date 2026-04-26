#!/usr/bin/env bash
# One-time setup: verify llama.cpp is installed and the model is ready on the Mac Mini.
#
# Run this once on the Mac Mini. After running, start the server with:
#   source configs/local-mac-m4.env && ./scripts/serve_consultant_llamacpp.sh
#
# Usage: bash scripts/mac_mini_setup.sh

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

echo -e "${BOLD}Mac Mini llama.cpp Remote Access Setup${NC}"
echo "========================================"
date
echo ""

# ── 1. Verify llama-server is installed ───────────────────────────────────────
if ! command -v llama-server &>/dev/null; then
    die "llama-server not found in PATH.\n  Install llama.cpp: brew install llama.cpp\n  Or build from source: https://github.com/ggml-org/llama.cpp"
fi
info "llama-server found: $(llama-server --version 2>&1 | head -1)"

# ── 2. Verify the GGUF model file exists ──────────────────────────────────────
MODEL_PATH="${CONSULTANT_MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
    die "CONSULTANT_MODEL_PATH is not set. Source your experiment .env first, e.g.:\n  set -a && source configs/local-mac-m4.env && set +a"
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    die "Model file not found: $MODEL_PATH\n  Download a .gguf from https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF"
fi
info "Model file found: $MODEL_PATH ($(du -sh "$MODEL_PATH" | cut -f1))"

# ── 3. Check macOS firewall ────────────────────────────────────────────────────
PORT="${CONSULTANT_PORT:-8080}"
warn "If the host PC can't reach port $PORT, check macOS firewall:"
warn "  System Settings → Network → Firewall → Options"
warn "  Make sure incoming connections on port $PORT are allowed."

# ── 4. Print this machine's address ──────────────────────────────────────────
MAC_HOSTNAME=$(scutil --get LocalHostName 2>/dev/null || echo "")
MAC_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "unknown")
MAC_ADDR="${MAC_HOSTNAME:+${MAC_HOSTNAME}.local}"
MAC_ADDR="${MAC_ADDR:-$MAC_IP}"
MODEL_NAME="${CONSULTANT_MODEL_NAME:-$(basename "$MODEL_PATH" .gguf)}"

echo ""
echo -e "${BOLD}Mac Mini address: $MAC_ADDR${NC}"
echo "(IP fallback: $MAC_IP)"
echo ""
echo "Start the consultant server on this Mac Mini:"
echo ""
echo "  set -a && source configs/local-mac-m4.env && set +a"
echo "  ./scripts/serve_consultant_llamacpp.sh"
echo ""
echo "From the host PC, set these before running the eval:"
echo ""
echo "  export CONSULTANT_BASE_URL=http://${MAC_ADDR}:${PORT}/v1"
echo "  export CONSULTANT_MODEL_NAME=${MODEL_NAME}"
echo "  export CONSULTANT_API_KEY=no-key"
echo ""
echo "Then run the evaluation:"
echo ""
echo "  ./scripts/run_eval.sh local"
echo ""

info "Setup check complete."
