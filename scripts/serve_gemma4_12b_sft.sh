#!/usr/bin/env bash
# Serve the merged Socratic-SFT Gemma 4 12B (Q8_0 GGUF) as an OpenAI-compatible API
# on the NVIDIA RTX 4000 Ada. Thin wrapper over scripts/serve_gemma4_31b.sh.
#
# The weight file is produced by the merge+convert pipeline (see the 12B PoC plan):
#   scripts/merge_lora_gemma4_sft.py --base google/gemma-4-12b-it \
#     --adapter outputs/sft-gemma4-12b-qlora/final --out outputs/sft-gemma4-12b-qlora/merged
#   bash scripts/convert_gemma4_12b_sft_to_gguf.sh
# The convert wrapper writes gemma-4-12B-kele-socratic-sft-Q8_0.gguf AND stages it
# into the weights dir below (matching GEMMA4_12B_SFT_WEIGHT_FILE) — no rename needed.
#
# Distinct alias "Gemma 4 12B SFT" so the eval orchestrator proves the SFT weights
# answered (gemma4-12b-sft-local.env). Stop the base server first — one model at a
# time on the 20 GB card.
#
# Endpoint: http://localhost:8080/v1/chat/completions
# Override the weights dir with GEMMA4_12B_WEIGHTS_DIR=/some/path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GEMMA4_12B_WEIGHTS_DIR="${GEMMA4_12B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${GEMMA4_12B_SFT_WEIGHT_FILE:-gemma-4-12B-kele-socratic-sft-Q8_0.gguf}"
MTP_FILE="${GEMMA4_12B_MTP_FILE:-MTP/gemma-4-12B-it-MTP-Q8_0.gguf}"

# MTP=1 attaches the same base-derived drafter as the base server. The drafter
# sees the SFT'd distribution → lower acceptance / smaller speedup than on base,
# but it stays lossless. Used by stage 3 when MTP wins the base on/off A/B.
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
  -a "Gemma 4 12B SFT" \
  -c "${GEMMA4_12B_CTX:-32768}" \
  "${MTP_ARGS[@]}" \
  "$@"
