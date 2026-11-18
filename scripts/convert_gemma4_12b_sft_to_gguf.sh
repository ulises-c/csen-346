#!/usr/bin/env bash
# Convert merged Gemma-4-12B SFT (HF BF16) → f16 GGUF → Q8_0 GGUF, then stage it
# where scripts/serve_gemma4_12b_sft.sh looks.
#
# Pipeline step 2+3 of 3 — assumes scripts/merge_lora_gemma4_sft.py already ran:
#   scripts/merge_lora_gemma4_sft.py --base google/gemma-4-12b-it \
#     --adapter outputs/sft-gemma4-12b-qlora/final --out outputs/sft-gemma4-12b-qlora/merged
#
# This is the 12B sibling of convert_gemma4_sft_to_gguf.sh. It exists so the
# output filename carries the 12B NAME_TAG (the 31B script hardcodes "31B"),
# matching serve_gemma4_12b_sft.sh's default GEMMA4_12B_SFT_WEIGHT_FILE — so no
# manual rename is needed at G6 of the PoC.
#
# Why Q8_0: the base teacher is served at UD-Q8_K_XL (Unsloth dynamic, not
# producible by llama-quantize). Q8_0 is the standard llama.cpp 8-bit quant at the
# same bit budget; at Q8 the quant-scheme delta vs the base is ~noise, keeping the
# base↔SFT comparison clean (the difference is the LoRA adapter, not the quant).
#
# Usage:
#   bash scripts/convert_gemma4_12b_sft_to_gguf.sh
#
# Override:
#   MERGED_DIR=... GGUF_DIR=... QUANT=Q6_K  bash scripts/convert_gemma4_12b_sft_to_gguf.sh
#   GEMMA4_12B_WEIGHTS_DIR=/some/path  bash scripts/convert_gemma4_12b_sft_to_gguf.sh  # stage target
#   NO_STAGE=1  bash scripts/convert_gemma4_12b_sft_to_gguf.sh                          # skip the cp

set -euo pipefail

# Resolve relative paths against the repo root so `cd "$LLAMA_CPP_DIR"` below
# doesn't break them. realpath -m allows paths whose final component doesn't
# exist yet (the GGUF_DIR/output files are about to be created).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MERGED_DIR="$(realpath -m "${MERGED_DIR:-$REPO_ROOT/outputs/sft-gemma4-12b-qlora/merged}")"
GGUF_DIR="$(realpath -m "${GGUF_DIR:-$REPO_ROOT/outputs/sft-gemma4-12b-qlora}")"
QUANT="${QUANT:-Q8_0}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/Documents/models/llama.cpp}"
CONVERT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
# Override QUANTIZE to a CPU-only build of llama-quantize: on the unstable NVIDIA
# box a CUDA recompile of ggml segfaults (nvcc Error 139 under load), and quantize
# is a CPU op that needs no GPU backend — see scripts/serve note / memory.
QUANTIZE="${QUANTIZE:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"

# KELE-tagged filenames — must NOT collide with base gemma-4-12b-it-*.gguf.
NAME_TAG="gemma-4-12B-kele-socratic-sft"
F16_GGUF="$GGUF_DIR/${NAME_TAG}-f16.gguf"
OUT_GGUF="$GGUF_DIR/${NAME_TAG}-${QUANT}.gguf"

# Where serve_gemma4_12b_sft.sh expects the final GGUF.
WEIGHTS_DIR="${GEMMA4_12B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -d "$MERGED_DIR" ]]; then
  echo "ERROR: Merged HF checkpoint not found at $MERGED_DIR" >&2
  echo "Run scripts/merge_lora_gemma4_sft.py --base google/gemma-4-12b-it first." >&2
  exit 1
fi

if [[ ! -f "$CONVERT" ]]; then
  echo "ERROR: convert_hf_to_gguf.py not found at $CONVERT" >&2
  exit 1
fi

if [[ ! -x "$QUANTIZE" ]]; then
  echo "ERROR: llama-quantize binary not found at $QUANTIZE" >&2
  exit 1
fi

mkdir -p "$GGUF_DIR"

if [[ -f "$OUT_GGUF" ]]; then
  echo "$OUT_GGUF already exists. Delete and re-run to overwrite."
  exit 0
fi

# ── Step 1: HF (BF16 merged) → f16 GGUF ───────────────────────────────────────
if [[ ! -f "$F16_GGUF" ]]; then
  echo "=== Step 1: HF → f16 GGUF ==="
  echo "Input:  $MERGED_DIR"
  echo "Output: $F16_GGUF"
  echo "(~5-8 min on CPU, ~24 GB write)"

  PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
  fi

  cd "$LLAMA_CPP_DIR"
  "$PYTHON" "$CONVERT" "$MERGED_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16
  cd - >/dev/null
fi

# ── Step 2: f16 GGUF → Q8_0 ───────────────────────────────────────────────────
echo
echo "=== Step 2: f16 GGUF → $QUANT ==="
echo "Input:  $F16_GGUF"
echo "Output: $OUT_GGUF"
echo "(~3-5 min on CPU)"

"$QUANTIZE" "$F16_GGUF" "$OUT_GGUF" "$QUANT"

# ── Step 3: stage where the serve wrapper looks ───────────────────────────────
if [[ "${NO_STAGE:-0}" != "1" ]]; then
  mkdir -p "$WEIGHTS_DIR"
  cp "$OUT_GGUF" "$WEIGHTS_DIR/"
  echo
  echo "Staged → $WEIGHTS_DIR/$(basename "$OUT_GGUF")"
  echo "serve_gemma4_12b_sft.sh will find it (GEMMA4_12B_SFT_WEIGHT_FILE default)."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo "=== Done ==="
ls -lh "$OUT_GGUF"
echo
echo "Intermediate f16 GGUF (~24 GB) preserved at:"
echo "  $F16_GGUF"
echo "Delete with:  rm $F16_GGUF"
echo
echo "Next:  make serve-gemma4-12b-sft   then   make eval-gemma4-12b-sft-smoke"
