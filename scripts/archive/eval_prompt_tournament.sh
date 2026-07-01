#!/usr/bin/env bash
# Phase 1 prompt-engineering tournament: 10 utilization cells × n=50 = 500 dlg.
#
# Composition: BERT consultant (results/state_classifier_v1/final) + Gemma 4 31B
# teacher + 10-shot teacher-side exemplars (the Phase 0 locked baseline). Each
# cell layers one env-var-gated utilization on top and runs n=50 to a per-cell
# output directory.
#
# Mutex constraint: at most one of #6/#7/#8 per cell — enforced by the cell
# list below (each is its own cell). Multi-call cells (#7 CoT, #8 N-best,
# #10 compressed history) auto-cap KELE_PARALLEL_WORKERS=2 to avoid swamping
# the llama-server's 6 KV slots.
#
# Usage:
#   ./scripts/eval_prompt_tournament.sh         # full 10-cell tournament
#   CELLS="1,9" ./scripts/eval_prompt_tournament.sh  # subset
#
# After the run, summarize with:
#   uv run python scripts/aggregate_tournament_leaderboard.py results/tournament-cell-*

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

BERT_CKPT="results/state_classifier_v1/final"
EXPERIMENT="gemma4-31b-local"
PORT=8080
LLAMA_URL="http://localhost:${PORT}"
EXPECTED_ALIAS="Gemma 4 31B"
N_DIALOGUES="${KELE_TOURNAMENT_N:-50}"

mkdir -p logs
TS=$(date -u +%Y-%m-%dT%H-%M-%S)
RUN_LOG="logs/tournament_${TS}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== Phase 1 prompt-engineering tournament ==="
echo "Started: $(date)"
echo "Baseline composition: BERT + Gemma 4 31B + 10-shot"
echo "n per cell:           ${N_DIALOGUES}"
echo

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -d "$BERT_CKPT" ]]; then
  echo "ERROR: BERT checkpoint missing at $BERT_CKPT" >&2
  exit 1
fi

# Ensure Gemma server is up (boot it if not, reuse if it is)
ready_check() {
  local resp
  resp=$(curl -s --max-time 3 "$LLAMA_URL/v1/models" 2>/dev/null) || return 1
  [[ -z "$resp" ]] && return 1
  echo "$resp" | grep -q '"Loading model"' && return 1
  echo "$resp" | grep -q "\"$EXPECTED_ALIAS\""
}

SERVER_PID=""
WE_BOOTED=false
cleanup_server() {
  if $WE_BOOTED && [[ -n "$SERVER_PID" ]]; then
    echo
    echo "Shutting down llama-server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT INT TERM

if ready_check; then
  echo "Reusing existing $EXPECTED_ALIAS server on $LLAMA_URL."
else
  echo "Booting Gemma 4 31B (cold load: 30-60s)..."
  bash "$ROOT/scripts/serve_gemma4_31b_q5.sh" > "logs/server_tournament_${TS}.log" 2>&1 &
  SERVER_PID=$!
  WE_BOOTED=true
  for i in $(seq 1 60); do
    if ready_check; then
      echo "Server ready (~$((i*2))s)"
      break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "ERROR: server PID died during boot" >&2
      tail -30 "logs/server_tournament_${TS}.log" >&2
      exit 1
    fi
    sleep 2
  done
  if ! ready_check; then
    echo "TIMEOUT waiting for server" >&2
    exit 1
  fi
fi

# ── Cell definitions ─────────────────────────────────────────────────────────
# Format: "CELL_NUMBER:CELL_NAME:ENV_VARS:WORKERS"
# WORKERS=4 for single-call cells, WORKERS=2 for multi-call cells (#7,#8,#10).
CELLS_ALL=(
  "9:persona:KELE_TEACHER_PERSONA=1:4"
  "1:length_budget:KELE_STAGE_LENGTH_BUDGET=1:4"
  "2:lexical_priors:KELE_LEXICAL_PRIORS=1:4"
  "6:format_retry:KELE_FORMAT_RETRY=1:4"
  "5:negative_exemplars:KELE_NEGATIVE_EXEMPLARS=1:4"
  "3:style_matched_exemplars:KELE_STYLE_MATCHED_EXEMPLARS=1:4"
  "4:per_state_exemplars:KELE_PER_STATE_EXEMPLARS=1:4"
  "10:compressed_history:KELE_COMPRESSED_HISTORY=1:2"
  "7:cot_scaffold:KELE_TEACHER_COT=1:2"
  "8:nbest_rerank:KELE_NBEST_RERANK=3:2"
)

# Optional CELLS env: comma-separated subset (e.g. CELLS="1,9,4")
if [[ -n "${CELLS:-}" ]]; then
  IFS=',' read -ra wanted <<< "$CELLS"
  filtered=()
  for c in "${CELLS_ALL[@]}"; do
    num="${c%%:*}"
    for w in "${wanted[@]}"; do
      if [[ "$num" == "$(echo "$w" | tr -d ' ')" ]]; then
        filtered+=("$c")
      fi
    done
  done
  CELLS_TO_RUN=("${filtered[@]}")
else
  CELLS_TO_RUN=("${CELLS_ALL[@]}")
fi

echo "Cells to run: ${#CELLS_TO_RUN[@]}"
for c in "${CELLS_TO_RUN[@]}"; do
  echo "  - cell ${c%%:*} (${c#*:*}: ${c%%:*})"
done | head -20
echo

# ── Run each cell ────────────────────────────────────────────────────────────
# Use an array so word splitting is explicit and shellcheck-safe; an empty
# array expands to nothing under "${INHIBIT[@]}", so the call site stays simple.
INHIBIT=()
if command -v systemd-inhibit &>/dev/null; then
  INHIBIT=(systemd-inhibit --what=sleep:idle --who=prompt_tournament --why=KELE-eval)
fi

OVERALL_START=$(date +%s)
for cell_def in "${CELLS_TO_RUN[@]}"; do
  IFS=':' read -ra parts <<< "$cell_def"
  CELL_NUM="${parts[0]}"
  CELL_NAME="${parts[1]}"
  CELL_ENV="${parts[2]}"
  CELL_WORKERS="${parts[3]}"
  OUT_DIR="results/tournament-cell-${CELL_NUM}-${CELL_NAME}"

  echo
  echo "─────────────────────────────────────────────────────────────"
  echo "Cell ${CELL_NUM} (${CELL_NAME}) → $OUT_DIR"
  echo "Env: KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=10 ${CELL_ENV}"
  echo "Workers: ${CELL_WORKERS}"
  echo "Start: $(date)"
  CELL_START=$(date +%s)

  # Skip if already complete (resume support)
  if [[ -f "$OUT_DIR/metrics_summary.json" ]]; then
    echo "✓ already complete, skipping (delete metrics_summary.json to re-run)"
    continue
  fi

  # Build env-var prefix for this cell (use `env` to set a variable whose name
  # is itself stored in a variable — bash can't do that with the inline
  # VAR=value syntax).
  env \
    PATH="$ROOT/.venv/bin:$PATH" \
    KELE_FEW_SHOT_TEACHER=1 \
    KELE_FEW_SHOT_N=10 \
    KELE_PARALLEL_WORKERS="$CELL_WORKERS" \
    "$CELL_ENV" \
    "${INHIBIT[@]}" uv run python -m src.project.kele \
      --experiment "$EXPERIMENT" \
      evaluate \
      --bert-consultant "$BERT_CKPT" \
      --output "$OUT_DIR" \
      --limit "$N_DIALOGUES"

  EXIT=$?
  CELL_END=$(date +%s)
  CELL_DUR=$((CELL_END - CELL_START))
  echo "End: $(date) (cell wall: $((CELL_DUR / 60))m $((CELL_DUR % 60))s, exit=$EXIT)"

  if [[ -f "$OUT_DIR/metrics_summary.json" ]]; then
    echo "Cell ${CELL_NUM} metrics:"
    cat "$OUT_DIR/metrics_summary.json"
  fi
done

OVERALL_END=$(date +%s)
TOTAL_MIN=$(( (OVERALL_END - OVERALL_START) / 60 ))
echo
echo "═════════════════════════════════════════════════════════════"
echo "Tournament complete in ${TOTAL_MIN} min."
echo
echo "Run the aggregator to view the leaderboard:"
echo "  PATH=\".venv/bin:\$PATH\" uv run python scripts/aggregate_tournament_leaderboard.py"
