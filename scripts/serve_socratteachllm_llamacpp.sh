#!/usr/bin/env bash
# Serve SocratTeachLLM (Q8_0 GGUF, llama.cpp) as an OpenAI-compatible API on the RTX 5090.
#
# Replaces the legacy vLLM path (scripts/serve_socratteachllm.sh, written by
# Max's teammate) which suffered chained bit-rot on the current CUDA/transformers
# stack (vLLM ABI mismatch, ChatGLM TorchScript libnvrtc, bitsandbytes
# tied_weights_keys; see docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md §Limitation).
# This script uses the same llama.cpp serving path we use for Gemma 4 31B and
# Qwen 3.6 27B — proven to work cleanly on this hardware.
#
# SocratTeachLLM = LoRA fine-tune of THUDM/glm-4-9b-chat (GLM4 architecture).
# Native context is 128K; we cap at 32K (plenty for KELE turns, leaves headroom
# for many parallel slots).
#
# Sizing on the 32 GB 5090 at Q8_0:
#   - Model:        ~10 GB
#   - KV @ 32K × 12 slots (Q4_0):  ~6.5 GB
#   - Compute:      ~1 GB
#   - Total:        ~17.5 GB   (~14 GB free — comfortable; CAN go higher slots)
#
# 12 parallel slots = 2× our Qwen 27B default (which is 6 slots at much larger
# context). Bumps end-to-end eval throughput when paired with KELE_PARALLEL_WORKERS
# in the eval orchestrators. Push to 16+ slots if you want more.
#
# Endpoint: http://localhost:8080/v1/chat/completions
# Alias:    SocratTeachLLM
#
# Convert HF → GGUF first (one-shot):
#   ./scripts/convert_socratteachllm_to_gguf.sh
#
# Override defaults with env vars (see body).

set -euo pipefail

# ── llama-server binary discovery (mirrors serve_qwen27b.sh) ──────────────────
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
WEIGHTS_DIR="${SOCRATTEACHLLM_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${SOCRATTEACHLLM_WEIGHT_FILE:-SocratTeachLLM-Q8_0.gguf}"
MODEL="$WEIGHTS_DIR/$WEIGHT_FILE"
ALIAS="${SOCRATTEACHLLM_ALIAS:-SocratTeachLLM}"
CONTEXT="${SOCRATTEACHLLM_CTX:-32768}"           # 32K; native is 128K
PARALLEL="${SOCRATTEACHLLM_PARALLEL:-12}"        # 12 slots — 2× Qwen 27B default
KV_QUANT="${SOCRATTEACHLLM_KV_QUANT:-q4_0}"
GPU_LAYERS="${SOCRATTEACHLLM_NGL:-99}"
HOST="${SOCRATTEACHLLM_HOST:-0.0.0.0}"
PORT="${SOCRATTEACHLLM_PORT:-8080}"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
  echo "Error: weight file not found at $MODEL" >&2
  echo "Run conversion first:  ./scripts/convert_socratteachllm_to_gguf.sh" >&2
  echo "Or override with:  SOCRATTEACHLLM_WEIGHT_FILE=…  $0" >&2
  exit 1
fi

echo "=== SocratTeachLLM (llama.cpp) ==="
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
