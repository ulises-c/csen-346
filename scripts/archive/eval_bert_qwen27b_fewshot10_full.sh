#!/usr/bin/env bash
# End-to-end orchestrator for the BERT + Qwen3.6-27B (Q5, think mode) +
# 10-shot exemplars integration. Third teacher arm of the Layer-2 mini-test
# campaign (alongside Gemma 4 31B and Qwen 35B-A3B).
#
# Qwen3.6-27B is the dense Qwen3.6 sibling of the 35B-A3B MoE — same family,
# different parameter activation pattern (dense 27B vs sparse 35B-A3B). With
# both BERT+consultant integration and 10-shot teacher exemplars, this run
# completes the local-headline triple comparison.
#
# n=50 wall clock projection: ~55-65 min based on Gemma and A3B baselines.
# Server: serve_qwen27b_q5_think.sh on port 8080, alias "Qwen 27B Q5".
# At Q5_K_XL + 256K context (= native n_ctx_train), llama-server uses ~25 GB
# VRAM (~7 GB headroom). The previous 416K config tripped NVRM Xid 8 watchdog
# lockups on 2026-05-22 — KV cache plus prompt cache pushed VRAM past the
# safe threshold.
#
# Usage:
#   ./scripts/eval_bert_qwen27b_fewshot10_full.sh
#
# Env overrides for variant runs:
#   OUT_DIR=results/t4-bert-qwen27b-n50-fixed \
#   BERT_CKPT=results/state-clf-qwen3.5-0.8b-lora/final \
#   LIMIT=50 bash scripts/eval_bert_qwen27b_fewshot10_full.sh
#
# Output: results/bert-consultant-fewshot10-qwen27b-full/

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

OUT_DIR="${OUT_DIR:-results/bert-consultant-fewshot10-qwen27b-full}"
BERT_CKPT="${BERT_CKPT:-results/state_classifier_v1/final}"
LIMIT="${LIMIT:-}"
EXPERIMENT="qwen27b-local"
PORT=8080
LLAMA_URL="http://localhost:${PORT}"
EXPECTED_ALIAS="Qwen 27B Q5"

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

echo "=== BERT + Qwen3.6-27B (Q5 think) + 10-shot full eval ==="
echo "Started: $(date)"
echo "Output:  $OUT_DIR"
echo "BERT:    $BERT_CKPT"
echo "Server:  $SERVER_LOG"
echo

# ── GPU sanity ────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "GPU VRAM free: ${VRAM_FREE} MiB"
  if [[ "$VRAM_FREE" =~ ^[0-9]+$ ]] && [[ "$VRAM_FREE" -lt 25500 ]]; then
    echo "WARN: less than 25.5 GB VRAM free — Qwen 27B Q5_K_XL needs ~25 GB at 256K ctx." >&2
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
  echo "Booting llama-server (cold load: 30-90s)..."
  bash "$ROOT/scripts/serve_qwen27b_q5_think.sh" > "$SERVER_LOG" 2>&1 &
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
echo "Experiment: $EXPERIMENT (BERT consultant + Qwen3.6-27B teacher + 10-shot)"
echo "Full test split: n=681 dialogues, ~4170 turns"
echo

INHIBIT=()
if command -v systemd-inhibit &>/dev/null; then
  INHIBIT=(systemd-inhibit --what=sleep:idle --who=bert_qwen27b_full --why=KELE-eval)
fi

# Parallel workers default to 1 until parallel-eval is validated end-to-end.
# Override via env, e.g.:  KELE_PARALLEL_WORKERS=4 bash scripts/eval_bert_qwen27b_fewshot10_full.sh
KELE_PARALLEL_WORKERS="${KELE_PARALLEL_WORKERS:-1}"
echo "Parallel workers: $KELE_PARALLEL_WORKERS (server -np must be ≥ this)"

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
