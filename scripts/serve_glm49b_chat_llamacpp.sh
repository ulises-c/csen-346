#!/usr/bin/env bash
# Serve GLM-4-9B-Chat (Q8_0 GGUF, llama.cpp) as an OpenAI-compatible API on the RTX 5090.
#
# THUDM/glm-4-9b-chat is the BASE model that SocratTeachLLM was LoRA-fine-tuned
# from. Mirrors scripts/serve_socratteachllm_llamacpp.sh exactly — same arch
# (ChatGLM4), same memory profile, same parallel-slot sizing — only the weight
# file and alias differ.
#
# Native context is 128K; we cap at 32K (plenty for KELE turns, leaves headroom
# for many parallel slots).
#
# Sizing on the 32 GB 5090 at Q8_0 (identical to STL):
#   - Model:        ~10 GB
#   - KV @ 32K × 12 slots (Q4_0):  ~6.5 GB
#   - Compute:      ~1 GB
#   - Total:        ~17.5 GB   (~14 GB free)
#
# Endpoint: http://localhost:8080/v1/chat/completions
# Alias:    GLM-4-9B-Chat-base
#
# Convert HF → GGUF first (one-shot):
#   ./scripts/convert_glm49b_chat_to_gguf.sh
#
# Override defaults with env vars (see body).

set -euo pipefail

# ── llama-server binary discovery ─────────────────────────────────────────────
if [[ -z "${LLAMA_SERVER:-}" ]]; then
  if [[ -x "$HOME/Github/llama.cpp/build/bin/llama-server" ]]; then
    LLAMA_SERVER="$HOME/Github/llama.cpp/build/bin/llama-server"
  elif [[ -x "$HOME/Documents/models/llama.cpp/build/bin/llama-server" ]]; then
    LLAMA_SERVER="$HOME/Documents/models/llama.cpp/build/bin/llama-server"
  elif command -v llama-server >/dev/null 2>&1; then
    LLAMA_SERVER="$(command -v llama-server)"   # e.g. a distro llama.cpp package
  else
    LLAMA_SERVER="$HOME/Documents/models/llama.cpp/build/bin/llama-server"
  fi
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "Error: llama-server not found at $LLAMA_SERVER" >&2
  echo "Build llama.cpp first — see ~/Documents/models/REFERENCE.md" >&2
  echo "Or override with:  LLAMA_SERVER=/path/to/llama-server $0 ..." >&2
  exit 1
fi

# ── Defaults (overridable by env vars) ────────────────────────────────────────
WEIGHTS_DIR="${GLM49B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${GLM49B_WEIGHT_FILE:-glm-4-9b-chat-Q8_0.gguf}"
MODEL="$WEIGHTS_DIR/$WEIGHT_FILE"
ALIAS="${GLM49B_ALIAS:-GLM-4-9B-Chat-base}"
CONTEXT="${GLM49B_CTX:-32768}"
PARALLEL="${GLM49B_PARALLEL:-12}"
KV_QUANT="${GLM49B_KV_QUANT:-q4_0}"
GPU_LAYERS="${GLM49B_NGL:-99}"
HOST="${GLM49B_HOST:-0.0.0.0}"
PORT="${GLM49B_PORT:-8080}"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
  echo "Error: weight file not found at $MODEL" >&2
  echo "Run conversion first:  ./scripts/convert_glm49b_chat_to_gguf.sh" >&2
  echo "Or override with:  GLM49B_WEIGHT_FILE=…  $0" >&2
  exit 1
fi

echo "=== GLM-4-9B-Chat-base (llama.cpp) ==="
echo "Binary:    $LLAMA_SERVER"
echo "Weights:   $MODEL"
echo "Alias:     $ALIAS"
echo "Context:   $CONTEXT  (per-slot share: ~$((CONTEXT / PARALLEL)))"
echo "Slots:     $PARALLEL  (unified KV, continuous batching)"
echo "KV quant:  $KV_QUANT"
echo "Endpoint:  http://localhost:${PORT}/v1/chat/completions"
echo "Test:      curl http://localhost:${PORT}/v1/models"
echo "---"

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  -a "$ALIAS" \
  -ngl "$GPU_LAYERS" \
  -c "$CONTEXT" \
  -np "$PARALLEL" \
  --kv-unified \
  -ctk "$KV_QUANT" \
  -ctv "$KV_QUANT" \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
