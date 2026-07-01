#!/usr/bin/env bash
# End-to-end orchestrator for the BERT × GLM-4-9B-Chat-base cells in the
# STL ablation campaign (mk/glm-4-test branch). Mirrors
# scripts/eval_bert_socratteachllm_fewshot10_full.sh exactly with the teacher
# pointed at GLM-4-9B-Chat-base served via llama.cpp on port 8080 instead of
# SocratTeachLLM.
#
# GLM-4-9B-Chat is the BASE model that SocratTeachLLM was LoRA-fine-tuned from.
# This eval is the missing half of the true fine-tune-contribution ablation:
# STL minus its own pre-trained base, at matched n=681 + matched consultant +
# matched fewshot10 settings.
#
# Serving: vLLM on port 8001 (NOT llama.cpp on 8080). The llama.cpp ChatGLM
# converter doesn't embed BPE merges, so llama-server fails to load any
# ChatGLM4 GGUF on this box (affects this base model + the existing
# SocratTeachLLM-Q8_0.gguf). vLLM serves HF weights directly via
# trust_remote_code — proven path from STL's recorded n=50 May runs.
#
# Variants (via env overrides):
#
#   (1) bert-fixed × GLM-4-9B-Chat-base · ZH · n=681
#       OUT_DIR=results/bert-fixed-bert-glm49b-chat-fewshot10-n681-fixed \
#       BERT_CKPT=results/state_classifier_v1/final \
#       DATASET_PATH=references/KELE/SocratDataset.json \
#       bash scripts/eval_bert_glm49b_chat_fewshot10_full.sh
#
#   (2) qwen3.5 × GLM-4-9B-Chat-base · ZH · n=681
#       OUT_DIR=results/t4-bert-glm49b-chat-fewshot10-n681-fixed \
#       BERT_CKPT=results/state-clf-qwen3.5-0.8b-lora/final \
#       DATASET_PATH=references/KELE/SocratDataset.json \
#       bash scripts/eval_bert_glm49b_chat_fewshot10_full.sh
#
# Prereqs:
#   1. Download weights (one-shot):  hf download THUDM/glm-4-9b-chat --local-dir ~/hf_models/glm-4-9b-chat
#   2. Convert weights (one-shot):  bash scripts/convert_glm49b_chat_to_gguf.sh
#   3. The script self-boots scripts/serve_glm49b_chat_llamacpp.sh below.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

OUT_DIR="${OUT_DIR:-results/bert-glm49b-chat-fewshot10-n681}"
BERT_CKPT="${BERT_CKPT:-results/state_classifier_v1/final}"
LIMIT="${LIMIT:-}"
EXPERIMENT="${EXPERIMENT:-glm49b-chat-local}"
PORT="${PORT:-8001}"
LLAMA_URL="http://localhost:${PORT}"
EXPECTED_ALIAS="${EXPECTED_ALIAS:-GLM-4-9B-Chat-base}"
SERVE_SCRIPT="${SERVE_SCRIPT:-scripts/serve_glm49b_chat.sh}"

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

echo "=== BERT × GLM-4-9B-Chat-base (vLLM) eval ==="
echo "Started: $(date)"
echo "Output:  $OUT_DIR"
echo "BERT:    $BERT_CKPT"
echo "Server:  $SERVER_LOG"
echo

# ── GPU sanity ────────────────────────────────────────────────────────────────
# vLLM weights ~18 GB bf16 + KV pool ~5 GB at 0.70 mem util on a 32 GB card.
if command -v nvidia-smi &>/dev/null; then
  VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "GPU VRAM free: ${VRAM_FREE} MiB"
  if [[ "$VRAM_FREE" =~ ^[0-9]+$ ]] && [[ "$VRAM_FREE" -lt 24000 ]]; then
    echo "WARN: less than 24 GB VRAM free — vLLM bf16 weights + KV pool need ~24 GB at 0.70 util." >&2
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
  echo "Booting vLLM server (cold load: 60-180s; vLLM compiles CUDA kernels)..."
  bash "$ROOT/$SERVE_SCRIPT" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  WE_BOOTED=true

  echo -n "Waiting for $EXPECTED_ALIAS to be ready "
  for i in $(seq 1 300); do
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
    echo " TIMEOUT after 600s"
    exit 1
  fi
fi

# ── Eval ─────────────────────────────────────────────────────────────────────
echo
echo "=== Eval ==="
echo "Time: $(date)"
echo "Experiment: $EXPERIMENT (BERT consultant + GLM-4-9B-Chat-base teacher + 10-shot)"
echo

INHIBIT=()
if command -v systemd-inhibit &>/dev/null; then
  INHIBIT=(systemd-inhibit --what=sleep:idle --who=bert_glm49b_full --why=KELE-eval)
fi

KELE_PARALLEL_WORKERS="${KELE_PARALLEL_WORKERS:-4}"
echo "Parallel workers: $KELE_PARALLEL_WORKERS (vLLM handles batching internally)"

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

if [[ -f "$OUT_DIR/metrics_summary.json" ]]; then
  echo
  echo "=== Final metrics ==="
  cat "$OUT_DIR/metrics_summary.json"
fi

echo
echo "=== Done ==="
echo "Run log: $RUN_LOG"
echo "Results: $OUT_DIR/"
