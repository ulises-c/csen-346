#!/usr/bin/env bash
# Generic llama.cpp OpenAI-compatible server for the Gemma 4 31B local experiment.
#
# Mirrors the launcher pattern from ~/Documents/models/scripts/serve.sh so a
# single llama-server build (and weight directory) can serve both the personal
# AI stack and KELE without duplication.
#
# Usage:
#   ./scripts/serve_gemma4_31b.sh -m <weight.gguf> [-a alias] [-c ctx] [-np slots] [-p port] [extra args...]
#
# Model-specific scripts (e.g. serve_gemma4_31b_q5.sh) call this with preset
# defaults; any extra args pass through to llama-server.
#
# Endpoint: http://localhost:<port>/v1/chat/completions

set -euo pipefail

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

# Defaults (overridable by model scripts or CLI flags) — match upstream serve.sh
MODEL=""
ALIAS=""
CONTEXT=65536
KV_QUANT="q4_0"
GPU_LAYERS=99
# Match KELE_PARALLEL_WORKERS=4 used by eval scripts. Idle slots are not free:
# llama.cpp reserves per-slot KV cache bookkeeping and accumulates per-slot
# context checkpoints (~200-270 MiB each, up to 32 per slot). With -np 6 and
# only 4 workers, the LCP-similarity router still spreads requests across 5-6
# slots over time, inflating checkpoint memory and contributing to the
# 2026-05-26 crash on t4-bert-gemma-fewshot10-n681. Pinning -np to KELE
# worker count keeps the cache concentrated and frees ~500 MiB headroom.
PARALLEL=4
HOST="0.0.0.0"
PORT=8080
# Auto-detect GPU backend: AMD (/dev/kfd present) → Vulkan0 (fastest on gfx1201),
# NVIDIA → CUDA0. Override: DEV=ROCm0, DEV=CUDA0, etc.
if [[ -z "${DEV:-}" ]]; then
  # Ask the binary which devices it was built for: a ROCm/HIP build names the
  # card "ROCm0", a Vulkan build "Vulkan0", a CUDA build "CUDA0".
  DEV="$("$LLAMA_SERVER" --list-devices 2>/dev/null |
    awk -F: '/^[[:space:]]+[A-Za-z]+[0-9]+:/ { gsub(/[[:space:]]/, "", $1); print $1; exit }' || true)"
  if [[ -z "$DEV" ]]; then
    if [[ -e /dev/kfd ]]; then
      DEV="Vulkan0"
    else
      DEV="CUDA0"
    fi
  fi
fi
# 2048 is the safe default on 32 GB VRAM with Gemma 4 31B Q5. The earlier 4096
# default OOM'd on 2026-05-26 (compute buffer at 4096 is ~6.7 GB, not the
# 1.3 GB the prior comment cited). 2048 drops the compute buffer to ~3.3 GB
# and was the working configuration for the t4-bert-fixed-gemma-fewshot10-n681
# run that completed cleanly 2026-05-27 (681/681 in 2h 37m).
UBATCH="${UBATCH:-2048}"
BATCH="${BATCH:-2048}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)             MODEL="$2";      shift 2 ;;
    -a|--alias)     ALIAS="$2";      shift 2 ;;
    -c)             CONTEXT="$2";    shift 2 ;;
    -np|--parallel) PARALLEL="$2";   shift 2 ;;
    -ctk|-ctv)      KV_QUANT="$2";   shift 2 ;;
    -ngl)           GPU_LAYERS="$2"; shift 2 ;;
    --host)         HOST="$2";       shift 2 ;;
    -p|--port)      PORT="$2";       shift 2 ;;
    -dev)           DEV="$2";        shift 2 ;;
    --ubatch-size)  UBATCH="$2";     shift 2 ;;
    -b|--batch-size) BATCH="$2";     shift 2 ;;
    *)              EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: no model specified. Use -m <path> or a model-specific script (serve_gemma4_31b_q5.sh)." >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Error: weight file not found at $MODEL" >&2
  echo "Download it first:  hf download unsloth/gemma-4-31b-it-GGUF $(basename "$MODEL") --local-dir \$(dirname \"$MODEL\")" >&2
  exit 1
fi

ALIAS_ARGS=()
if [[ -n "$ALIAS" ]]; then
  ALIAS_ARGS=(-a "$ALIAS")
fi

echo "=== Gemma 4 31B (llama.cpp) ==="
echo "Binary:    $LLAMA_SERVER"
echo "Weights:   $MODEL"
echo "Alias:     ${ALIAS:-<none>}"
echo "Context:   $CONTEXT  (per-slot: ~$((CONTEXT / PARALLEL)))"
echo "Slots:     $PARALLEL  (unified KV, continuous batching)"
echo "KV quant:  ${KV_QUANT}"
echo "Device:    ${DEV}"
echo "Batch:     ${BATCH} / Ubatch: ${UBATCH}"
echo "Endpoint:  http://localhost:${PORT}/v1/chat/completions"
echo "Test:      curl http://localhost:${PORT}/v1/models"
echo "---"

# Recent llama.cpp builds detect that Gemma 4's chat template supports thinking
# and turn it on (--reasoning auto). The teacher then writes about 600 reasoning
# tokens before each reply of about 35, which made a 50-dialogue run on the
# R9700 six times slower. The archived runs' wall times (681 dialogues in 32
# minutes on a 5090) are only consistent with thinking off, so pin it off.
# Override with LLAMA_ARG_REASONING=auto. Builds without the option ignore it.
export LLAMA_ARG_REASONING="${LLAMA_ARG_REASONING:-off}"

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  "${ALIAS_ARGS[@]}" \
  -ngl "$GPU_LAYERS" \
  -c "$CONTEXT" \
  -np "$PARALLEL" \
  --kv-unified \
  -ctk "$KV_QUANT" \
  -ctv "$KV_QUANT" \
  -fa 1 \
  -dev "$DEV" \
  -b "$BATCH" \
  --ubatch-size "$UBATCH" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
