#!/usr/bin/env bash
# End-to-end orchestrator for the BERT + Qwen 35B-A3B + 10-shot exemplars
# integration at n=681 (full test split).
#
# Phase 0.5 of the post-BERT-baseline plan (docs/PROMPT_ENGINEERING_PLAN.md):
# validate the teacher choice for the prompt-engineering tournament. The
# n=50 reference is BERT + A3B + 10-shot = 48.19% / 35.57 R-1 (Pareto win
# axes vs locked baseline). Standalone A3B beat standalone Gemma at full
# scale by +7.31 state-acc, driven by schema-fallback differences on the
# consultant path. The integration architecturally removes the schema-
# fallback dependency, so the teacher comparison is genuinely open at scale.
#
# n=50 wall clock was 60 min @ ~50 dlg/hr. Full run projects ~13-14 h.
#
# Usage:
#   ./scripts/eval_bert_a3b_fewshot10_full.sh
#
# Output: results/bert-consultant-fewshot10-a3b-full/

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

# Env overrides for variant runs (mini, T4 classifier, etc.):
#   OUT_DIR=results/t4-bert-a3b-n50 BERT_CKPT=results/state-clf-qwen3.5-0.8b-lora/final \
#     LIMIT=50 bash scripts/eval_bert_a3b_fewshot10_full.sh
OUT_DIR="${OUT_DIR:-results/bert-consultant-fewshot10-a3b-full}"
BERT_CKPT="${BERT_CKPT:-results/state_classifier_v1/final}"
LIMIT="${LIMIT:-}"
EXPERIMENT="qwen35b-a3b-local"
PORT=8080
LLAMA_URL="http://localhost:${PORT}"
EXPECTED_ALIAS="Qwen 35B A3B"

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

echo "=== BERT + Qwen 35B-A3B (think) + 10-shot full eval ==="
echo "Started: $(date)"
echo "Output:  $OUT_DIR"
echo "BERT:    $BERT_CKPT"
echo "Server:  $SERVER_LOG"
echo

# ── GPU sanity ────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "GPU VRAM free: ${VRAM_FREE} MiB"
  if [[ "$VRAM_FREE" =~ ^[0-9]+$ ]] && [[ "$VRAM_FREE" -lt 30000 ]]; then
    echo "WARN: less than 30 GB VRAM free — server may OOM." >&2
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
  # IMPORTANT: use the _think variant; the plain serve_qwen35b_a3b.sh has
  # --reasoning off baked in for the no-think tournament work.
  echo "Booting llama-server (think mode; cold load: 30-90s)..."
  bash "$ROOT/scripts/serve_qwen35b_a3b_think.sh" > "$SERVER_LOG" 2>&1 &
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
echo "Experiment: $EXPERIMENT (BERT consultant + A3B-think teacher + 10-shot)"
echo "Full test split: n=681 dialogues, ~4170 turns"
echo

INHIBIT=()
if command -v systemd-inhibit &>/dev/null; then
  INHIBIT=(systemd-inhibit --what=sleep:idle --who=bert_a3b_full --why=KELE-eval)
fi

# Parallel workers default to 1 until parallel-eval is validated end-to-end.
# Override via env, e.g.:  KELE_PARALLEL_WORKERS=4 bash scripts/eval_bert_a3b_fewshot10_full.sh
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
