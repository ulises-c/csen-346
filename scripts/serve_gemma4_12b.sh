#!/usr/bin/env bash
# Serve Gemma 4 12B-it (Unsloth UD-Q8_K_XL GGUF) as an OpenAI-compatible API on
# the NVIDIA RTX 4000 Ada (20 GB). Thin wrapper over the generic engine
# scripts/serve_gemma4_31b.sh (auto-selects DEV=CUDA0 when /dev/kfd is absent).
#
# Context is 32K (not the 31B's 150K): the 20 GB card holds the ~13.6 GB Q8 weights
# + KV + the in-process Qwen3.5 consultant, and KELE turns are <10K tokens anyway.
#
# MTP (multi-token prediction, llama.cpp PR #23398) — lossless speculative decode.
# Set MTP=1 to attach the base-derived drafter for a speed A/B. The drafter needs a
# llama.cpp build that includes PR #23398 (arch gemma4-assistant); stock builds
# cannot load it. MTP forces f16 KV cache: the PR's quantized-KV path (-ctk q8_0)
# initially showed 0% draft acceptance.
#
# Download once:
#   hf download unsloth/gemma-4-12b-it-GGUF gemma-4-12b-it-UD-Q8_K_XL.gguf \
#     --local-dir ~/Documents/models/weights
#   hf download unsloth/gemma-4-12b-it-GGUF MTP/gemma-4-12B-it-MTP-Q8_0.gguf \
#     --local-dir ~/Documents/models/weights
#
# Endpoint: http://localhost:8080/v1/chat/completions
# Override the weights dir with GEMMA4_12B_WEIGHTS_DIR=/some/path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GEMMA4_12B_WEIGHTS_DIR="${GEMMA4_12B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${GEMMA4_12B_WEIGHT_FILE:-gemma-4-12b-it-UD-Q8_K_XL.gguf}"
MTP_FILE="${GEMMA4_12B_MTP_FILE:-MTP/gemma-4-12B-it-MTP-Q8_0.gguf}"

# GEMMA4_12B_KV=f16 forces the KV cache type without MTP — needed for 1:1
# MTP on/off comparisons, since MTP forces f16 below while the engine default
# is q4_0 (a second variable otherwise).
KV_ARGS=()
if [[ -n "${GEMMA4_12B_KV:-}" ]]; then
  KV_ARGS=(-ctk "$GEMMA4_12B_KV" -ctv "$GEMMA4_12B_KV")
fi

MTP_ARGS=()
if [[ "${MTP:-0}" == "1" ]]; then
  mtp_path="$GEMMA4_12B_WEIGHTS_DIR/$MTP_FILE"
  if [[ ! -f "$mtp_path" ]]; then
    printf 'error: MTP=1 but drafter not found at %s\n' "$mtp_path" >&2
    printf '       hf download unsloth/gemma-4-12b-it-GGUF %s --local-dir %s\n' \
      "$MTP_FILE" "$GEMMA4_12B_WEIGHTS_DIR" >&2
    exit 1
  fi
  MTP_ARGS=(
    --spec-type draft-mtp
    --spec-draft-model "$mtp_path"
    --spec-draft-n-max "${GEMMA4_12B_MTP_NMAX:-4}"
    -ctk f16 -ctv f16
  )
fi

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$GEMMA4_12B_WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Gemma 4 12B" \
  -c "${GEMMA4_12B_CTX:-32768}" \
  "${KV_ARGS[@]}" \
  "${MTP_ARGS[@]}" \
  "$@"
