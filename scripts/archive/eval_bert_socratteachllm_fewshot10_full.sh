#!/usr/bin/env bash
# End-to-end orchestrator for the BERT × SocratTeachLLM cells in the master
# 4-cell bilingual probe (DONE 2026-05-23 PM — see docs/BILINGUAL_PROBE_RESULTS.md
# "STL bilingual arm" §; reused for the clean-probe synthetic eval). Mirrors
# scripts/eval_bert_gemma_fewshot10_full.sh exactly, with
# the teacher pointed at the local vLLM-served SocratTeachLLM (port 8001)
# instead of Gemma 4 31B.
#
# The llama.cpp serving path (port 8080, scripts/serve_socratteachllm_llamacpp.sh)
# is currently blocked on the chatglm GGUF converter (BPE merges issue): the chatglm.py converter pulls BPE merges
# unconditionally and SocratTeachLLM only ships tiktoken (no merges.txt). vLLM
# trust-remote-code path works directly. Switch SERVE_SCRIPT once that lands.
#
# This eval orchestrates four expected variants via env overrides:
#
#   (1) bert-fixed × SocratTeachLLM · ZH · n=50
#       OUT_DIR=results/bert-fixed-bert-socratteachllm-fewshot10-n50-fixed \
#       BERT_CKPT=results/state_classifier_v1/final \
#       LIMIT=50 SAMPLE_SEED=42 \
#       bash scripts/eval_bert_socratteachllm_fewshot10_full.sh
#
#   (2) bert-fixed × SocratTeachLLM · EN · n=50
#       OUT_DIR=results/bert-fixed-bert-socratteachllm-fewshot10-EN-n50-fixed \
#       BERT_CKPT=results/state_classifier_v1/final \
#       LIMIT=50 SAMPLE_SEED=42 \
#       DATASET_PATH=references/KELE-EN/SocratDataset.json \
#       bash scripts/eval_bert_socratteachllm_fewshot10_full.sh
#
#   (3) qwen3.5 × SocratTeachLLM · ZH · n=50
#       OUT_DIR=results/t4-bert-socratteachllm-fewshot10-n50-fixed \
#       BERT_CKPT=results/state-clf-qwen3.5-0.8b-lora/final \
#       LIMIT=50 SAMPLE_SEED=42 \
#       bash scripts/eval_bert_socratteachllm_fewshot10_full.sh
#
#   (4) qwen3.5 × SocratTeachLLM · EN · n=50
#       (combine 2 + 3 env overrides)
#
# Prereqs:
#   1. Convert weights (one-shot):  bash scripts/convert_socratteachllm_to_gguf.sh
#   2. The script self-boots scripts/serve_socratteachllm_llamacpp.sh below.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

OUT_DIR="${OUT_DIR:-results/bert-socratteachllm-fewshot10-n50}"
BERT_CKPT="${BERT_CKPT:-results/state_classifier_v1/final}"
LIMIT="${LIMIT:-}"
EXPERIMENT="socratteachllm-local"
# Port 8001 = vLLM default (per scripts/serve_socratteachllm.sh).
# llama.cpp path on port 8080 is blocked on the chatglm GGUF converter (BPE merges issue) (missing BPE merges
# in chatglm GGUF conversion); using vLLM until that's fixed.
PORT="${PORT:-8001}"
LLAMA_URL="http://localhost:${PORT}"
EXPECTED_ALIAS="SocratTeachLLM"
# Which serve script to boot if no server running. Defaults to vLLM
# (working path); set SERVE_SCRIPT to the llama.cpp variant once the chatglm GGUF converter (BPE merges issue) lands.
SERVE_SCRIPT="${SERVE_SCRIPT:-scripts/serve_socratteachllm.sh}"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -d "$BERT_CKPT" ]]; then
  echo "ERROR: BERT checkpoint missing at $BERT_CKPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
TS=$(date -u +%Y-%m-%dT%H-%M-%S)
RUN_LOG="$OUT_DIR/run_${TS}.log"
SERVER_LOG="$OUT_DIR/server_${TS}.log"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== BERT × SocratTeachLLM (llama.cpp Q8_0) eval ==="
echo "Started: $(date)"
echo "Output:  $OUT_DIR"
echo "BERT:    $BERT_CKPT"
echo "Server:  $SERVER_LOG"
echo

# ── GPU sanity ────────────────────────────────────────────────────────────────
# SocratTeachLLM Q8_0 + 12 parallel slots @ 32K ctx uses ~17.5 GB. Comfortable.
if command -v nvidia-smi &>/dev/null; then
  VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "GPU VRAM free: ${VRAM_FREE} MiB"
  if [[ "$VRAM_FREE" =~ ^[0-9]+$ ]] && [[ "$VRAM_FREE" -lt 18000 ]]; then
    echo "WARN: less than 18 GB VRAM free — SocratTeachLLM Q8_0 needs ~17.5 GB at 32K × 12 slots." >&2
  fi
fi

# ── Server lifecycle ─────────────────────────────────────────────────────────
SERVER_PID=""
WE_BOOTED=false

ready_check() {
  local resp
  resp=$(curl -s --max-time 3 "$LLAMA_URL/v1/models" 2>/dev/null) || return 1
  [[ -z "$resp" ]] && return 1
  echo "$resp" | grep -q '"Loading model"' && return 1
  echo "$resp" | grep -q "\"$EXPECTED_ALIAS\""
}

cleanup_server() {
  if $WE_BOOTED && [[ -n "$SERVER_PID" ]]; then
    echo
    echo "Shutting down llama-server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server stopped."
        return 0
      fi
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    echo "Server force-killed."
  fi
}
trap cleanup_server EXIT INT TERM

if ready_check; then
  echo "Reusing existing $EXPECTED_ALIAS server on $LLAMA_URL."
else
  echo "Booting teacher server via $SERVE_SCRIPT (cold load: 30-180s depending on backend)..."
  bash "$ROOT/$SERVE_SCRIPT" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  WE_BOOTED=true

  echo -n "Waiting for $EXPECTED_ALIAS to be ready "
  for i in $(seq 1 180); do
    if ready_check; then
      echo " ready (~$((i*2))s)"
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo " FAILED — server PID died."
      tail -30 "$SERVER_LOG" >&2
      exit 1
    fi
    echo -n "."
    sleep 2
  done
  if ! ready_check; then
    echo " TIMEOUT after 360s"
    exit 1
  fi
fi

# ── Eval ─────────────────────────────────────────────────────────────────────
echo
echo "=== Eval ==="
echo "Time: $(date)"
echo "Experiment: $EXPERIMENT (BERT consultant + SocratTeachLLM teacher + 10-shot)"
echo "Full test split: n=681 dialogues, ~4170 turns"
echo

INHIBIT=()
if command -v systemd-inhibit &>/dev/null; then
  INHIBIT=(systemd-inhibit --what=sleep:idle --who=bert_stl_full --why=KELE-eval)
fi

# Parallel workers default to 4 here (vs the usual 1 for big teachers) because
# SocratTeachLLM is small (9B) and the server is configured with 12 slots —
# we can afford to fan out and amortize the per-request overhead. Override
# via env, e.g.:  KELE_PARALLEL_WORKERS=8 bash scripts/eval_bert_socratteachllm_fewshot10_full.sh
KELE_PARALLEL_WORKERS="${KELE_PARALLEL_WORKERS:-4}"
echo "Parallel workers: $KELE_PARALLEL_WORKERS (server -np must be ≥ this; default serve script sets 12)"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
  echo "Limit: $LIMIT dialogues"
fi
if [[ -n "${SAMPLE_SEED:-}" ]]; then
  LIMIT_ARGS+=(--sample-seed "$SAMPLE_SEED")
  echo "Sample seed: $SAMPLE_SEED (random subsample, not first-N-by-ID)"
fi
if [[ -n "${DATASET_PATH:-}" ]]; then
  LIMIT_ARGS+=(--dataset-path "$DATASET_PATH")
  echo "Dataset: $DATASET_PATH"
fi

PATH="$ROOT/.venv/bin:$PATH" \
KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=10 \
KELE_PARALLEL_WORKERS="$KELE_PARALLEL_WORKERS" \
  "${INHIBIT[@]}" uv run python -m src.project.kele \
    --experiment "$EXPERIMENT" \
    evaluate \
    --bert-consultant "$BERT_CKPT" \
    "${LIMIT_ARGS[@]}" \
    --output "$OUT_DIR"

EVAL_EXIT=$?
echo
echo "Eval finished (exit=$EVAL_EXIT): $(date)"

# ── Result summary ───────────────────────────────────────────────────────────
if [[ -f "$OUT_DIR/metrics_summary.json" ]]; then
  echo
  echo "=== Final metrics ==="
  cat "$OUT_DIR/metrics_summary.json"
fi

echo
echo "=== Done ==="
echo "Run log: $RUN_LOG"
echo "Results: $OUT_DIR/"
