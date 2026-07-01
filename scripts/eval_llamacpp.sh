#!/usr/bin/env bash
# Parameterized orchestrator for single-server llama.cpp evals (teacher = consultant
# on port 8080, dual-role). Boots the model's serve script if not already up, runs
# KELE eval, compares against the gpt-4o-2024-11-20 baseline. Crash-safe via kele.py's
# per-item resume.
#
# Replaces the per-model eval_qwen27b.sh / eval_qwen35b_a3b.sh / eval_gemma4_31b.sh /
# eval_gemma4_26b_a4b.sh / eval_qwopus35b_a3b.sh family — those are config rows now,
# not separate scripts. See the MODEL CONFIG table below.
#
# Usage:
#   ./scripts/eval_llamacpp.sh <model> <smoke|mini|full|n50> [flags]
#
#   <model>: qwen27b | qwen35b-a3b | qwopus35b-a3b | gemma4-31b | gemma4-26b-a4b
#   modes:   smoke (n=5)  mini (n=25)  full (n=681)  n50 (n=50)
#   flags:   --think        use the model's thinking-mode serve script
#            --nothink      no-think prompt mode (Qwen only; -nothink experiment+suffix)
#            --unified      single-call fusion architecture (-unified suffix)
#            --suffix NAME  extra -NAME suffix on the output dir
#            --keep-server  leave a server we booted running on exit
#            --no-compare   skip the baseline comparison
#
# Examples:
#   ./scripts/eval_llamacpp.sh qwen27b smoke
#   ./scripts/eval_llamacpp.sh qwen27b smoke --think
#   ./scripts/eval_llamacpp.sh gemma4-31b full --unified
#
# Server policy: if a server already serves the expected alias on the port, reuse it
# and NEVER kill it. If we boot one, tear it down on exit unless --keep-server.
# Only one model fits at a time on the 5090's 32 GB VRAM — stop any other server first.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ── Args ──────────────────────────────────────────────────────────────────────
usage() {
  echo "Usage: $0 <qwen27b|qwen35b-a3b|qwopus35b-a3b|gemma4-31b|gemma4-26b-a4b> {smoke|mini|full|n50} [--think] [--nothink] [--unified] [--suffix NAME] [--keep-server] [--no-compare]" >&2
}

MODEL="${1:-}"
shift || true
case "$MODEL" in
  qwen27b|qwen35b-a3b|qwopus35b-a3b|gemma4-31b|gemma4-26b-a4b) ;;
  -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
  *) echo "Unknown or missing model: '$MODEL'" >&2; usage; exit 1 ;;
esac

MODE=""
KEEP_SERVER=false
DO_COMPARE=true
NOTHINK=false
UNIFIED=false
THINK=false
EXTRA_SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    smoke|mini|full|n50) MODE="$1"; shift ;;
    --keep-server)   KEEP_SERVER=true; shift ;;
    --no-compare)    DO_COMPARE=false; shift ;;
    --nothink)       NOTHINK=true; shift ;;
    --think)         THINK=true; shift ;;
    --unified)       UNIFIED=true; shift ;;
    --suffix)        EXTRA_SUFFIX="-$2"; shift 2 ;;
    -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then usage; exit 1; fi

# ── MODEL CONFIG ──────────────────────────────────────────────────────────────
# One row per model: serve script (default + thinking-mode), expected llama.cpp
# alias, GGUF file, base experiment name, VRAM-free warning threshold (MiB), and
# whether the model has a thinking-mode (Qwen only — Gemma rejects --think/--nothink).
WEIGHTS_DIR="${LLAMACPP_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
case "$MODEL" in
  qwen27b)
    SERVE_DEFAULT="serve_qwen27b_q5.sh";    SERVE_THINK="serve_qwen27b_q5_think.sh"
    EXPECTED_ALIAS="Qwen 27B Q5";           WEIGHT_FILE="Qwen3.6-27B-UD-Q5_K_XL.gguf"
    EXPERIMENT_BASE="qwen27b-local";        VRAM_MIN=27000;  THINK_CAPABLE=true ;;
  qwen35b-a3b)
    SERVE_DEFAULT="serve_qwen35b_a3b.sh";   SERVE_THINK="serve_qwen35b_a3b_think.sh"
    EXPECTED_ALIAS="Qwen 35B A3B";          WEIGHT_FILE="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    EXPERIMENT_BASE="qwen35b-a3b-local";    VRAM_MIN=27000;  THINK_CAPABLE=true ;;
  qwopus35b-a3b)
    SERVE_DEFAULT="serve_qwopus35b_a3b_think.sh"; SERVE_THINK="serve_qwopus35b_a3b_think.sh"
    EXPECTED_ALIAS="Qwopus 35B A3B";        WEIGHT_FILE="Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf"
    EXPERIMENT_BASE="qwopus35b-a3b-local";  VRAM_MIN=27000;  THINK_CAPABLE=true ;;
  gemma4-31b)
    SERVE_DEFAULT="serve_gemma4_31b_q5.sh"; SERVE_THINK=""
    EXPECTED_ALIAS="Gemma 4 31B";           WEIGHT_FILE="gemma-4-31B-it-UD-Q5_K_XL.gguf"
    EXPERIMENT_BASE="gemma4-31b-local";     VRAM_MIN=26500;  THINK_CAPABLE=false ;;
  gemma4-26b-a4b)
    SERVE_DEFAULT="serve_gemma4_26b_a4b.sh"; SERVE_THINK=""
    EXPECTED_ALIAS="Gemma 4 26B A4B";       WEIGHT_FILE="gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
    EXPERIMENT_BASE="gemma4-26b-a4b-local"; VRAM_MIN=30500;  THINK_CAPABLE=false ;;
esac

if ! $THINK_CAPABLE && $NOTHINK; then
  echo "Error: --nothink is not applicable to $EXPECTED_ALIAS (no thinking-mode equivalent)." >&2
  exit 1
fi
if ! $THINK_CAPABLE && $THINK; then
  echo "Error: --think is not applicable to $EXPECTED_ALIAS (no thinking-mode equivalent)." >&2
  exit 1
fi

SERVE_SCRIPT="$SERVE_DEFAULT"
if $THINK; then SERVE_SCRIPT="$SERVE_THINK"; fi

# ── Mode config ───────────────────────────────────────────────────────────────
# Output dir suffix order: -unified first, -nothink second, then --suffix value.
EXPERIMENT="$EXPERIMENT_BASE"
SUFFIX=""
if $UNIFIED; then SUFFIX="${SUFFIX}-unified"; fi
if $NOTHINK; then EXPERIMENT="${EXPERIMENT_BASE}-nothink"; SUFFIX="${SUFFIX}-nothink"; fi

case "$MODE" in
  smoke) N=5;   OUT_DIR="results/${EXPERIMENT_BASE}-smoke${SUFFIX}${EXTRA_SUFFIX}"; SUBCMD="test"     ;;
  mini)  N=25;  OUT_DIR="results/${EXPERIMENT_BASE}-mini${SUFFIX}${EXTRA_SUFFIX}";  SUBCMD="test"     ;;
  n50)   N=50;  OUT_DIR="results/${EXPERIMENT_BASE}-n50${SUFFIX}${EXTRA_SUFFIX}";   SUBCMD="test"     ;;
  full)  N=0;   OUT_DIR="results/${EXPERIMENT_BASE}${SUFFIX}${EXTRA_SUFFIX}";       SUBCMD="evaluate" ;;
esac

# ── Constants / env-overridable paths ─────────────────────────────────────────
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/Documents/models/llama.cpp/build/bin/llama-server}"
PORT="${PORT:-8080}"
LLAMA_URL="http://localhost:${PORT}"
CONFIG_FILE="configs/${EXPERIMENT}.env"
BASELINE_DIR="results/baseline"

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo "=== $EXPECTED_ALIAS Eval Orchestrator ==="
echo "Mode:      $MODE  (n=$N, subcmd=$SUBCMD)"
echo "Experiment:$EXPERIMENT"
echo "Output:    $OUT_DIR"
echo "Serve:     $SERVE_SCRIPT"
echo "Compare:   $DO_COMPARE  (vs $BASELINE_DIR)"
echo "Keep srv:  $KEEP_SERVER"
echo "Think:     $THINK   Nothink: $NOTHINK   Unified: $UNIFIED"
echo "---"

preflight_check() {
  local name="$1" path="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: $name not found at $path" >&2
    exit 1
  fi
}

preflight_check "llama-server binary" "$LLAMA_SERVER"
preflight_check "weight file"         "$WEIGHTS_DIR/$WEIGHT_FILE"
preflight_check "config"              "$CONFIG_FILE"
preflight_check "serve script"        "$ROOT/scripts/$SERVE_SCRIPT"

if $DO_COMPARE && [[ ! -f "$BASELINE_DIR/metrics_summary.json" ]]; then
  echo "WARN: $BASELINE_DIR/metrics_summary.json missing — disabling comparison." >&2
  DO_COMPARE=false
fi

# GPU VRAM sanity (informational)
if command -v nvidia-smi &>/dev/null; then
  VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  echo "GPU VRAM free: ${VRAM_FREE} MiB"
  if [[ "$VRAM_FREE" =~ ^[0-9]+$ ]] && [[ "$VRAM_FREE" -lt "$VRAM_MIN" ]]; then
    echo "WARN: less than ${VRAM_MIN} MiB VRAM free — server may OOM." >&2
  fi
fi

# ── Run dir + log ─────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
TS=$(date -u +%Y-%m-%dT%H-%M-%S)
RUN_LOG="$OUT_DIR/run_${TS}.log"
SERVER_LOG="$OUT_DIR/server_${TS}.log"

# Mirror everything to the run log (process substitution preserves exit codes)
exec > >(tee -a "$RUN_LOG") 2>&1
echo "Run log: $RUN_LOG"
echo "Started: $(date)"
echo

# ── Server lifecycle ─────────────────────────────────────────────────────────
SERVER_PID=""
WE_BOOTED=false

probe_server() {
  curl -s --max-time 3 "$LLAMA_URL/v1/models" 2>/dev/null
}

# Quiet check: returns 0 if server serves the expected alias and is ready
# (model fully loaded — distinguish from the transient "Loading model" 503).
ready_check() {
  local resp
  resp=$(probe_server) || return 1
  [[ -z "$resp" ]] && return 1
  echo "$resp" | grep -q '"Loading model"' && return 1
  echo "$resp" | grep -q "\"$EXPECTED_ALIAS\""
}

# Loud check: prints diagnostic if a different model is being served
verify_alias() {
  local resp
  resp=$(probe_server)
  [[ -z "$resp" ]] && return 1
  if echo "$resp" | grep -q "\"$EXPECTED_ALIAS\""; then
    return 0
  fi
  echo "WARN: server on $LLAMA_URL serves a different model than '$EXPECTED_ALIAS':" >&2
  echo "$resp" >&2
  return 1
}

cleanup_server() {
  if $WE_BOOTED && [[ -n "$SERVER_PID" ]]; then
    if $KEEP_SERVER; then
      echo
      echo "Leaving llama-server (PID $SERVER_PID) running per --keep-server."
      echo "  Stop manually with:  kill $SERVER_PID"
      return 0
    fi
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
  echo "Reusing existing llama-server on $LLAMA_URL (alias '$EXPECTED_ALIAS' ready)."
elif probe_server >/dev/null && ! verify_alias; then
  echo "ERROR: server on $LLAMA_URL does not serve '$EXPECTED_ALIAS'." >&2
  echo "       Stop it first or override PORT=<other> $0 $MODEL $MODE" >&2
  exit 1
else
  if probe_server >/dev/null; then
    echo "Existing server on $LLAMA_URL is loading the model — waiting for it..."
  else
    echo "Booting llama-server (cold load: 30-90 s for a ~20 GB GGUF)..."
    echo "Server log: $SERVER_LOG"
    bash "$ROOT/scripts/$SERVE_SCRIPT" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    WE_BOOTED=true
  fi

  echo -n "Waiting for $EXPECTED_ALIAS to be ready "
  for i in $(seq 1 180); do
    if ready_check; then
      echo " ready (~$((i*2))s)"
      break
    fi
    if $WE_BOOTED && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo " FAILED — server process exited"
      echo "Last 30 lines of server log:" >&2
      tail -30 "$SERVER_LOG" >&2
      exit 1
    fi
    echo -n "."
    sleep 2
  done
  if ! ready_check; then
    echo " TIMEOUT after 360 s" >&2
    exit 1
  fi
fi

# ── Run eval ─────────────────────────────────────────────────────────────────
echo
echo "=== Eval ==="
echo "Time: $(date)"

INHIBIT=""
if command -v systemd-inhibit &>/dev/null; then
  INHIBIT="systemd-inhibit --what=sleep:idle --who=eval_llamacpp.sh --why=KELE-eval"
fi

EXTRA_KELE_ARGS=()
if $UNIFIED; then EXTRA_KELE_ARGS+=(--unified); fi

if [[ "$SUBCMD" == "test" ]]; then
  $INHIBIT uv run python -m src.project.kele \
    --experiment "$EXPERIMENT" test --n "$N" --output "$OUT_DIR" \
    "${EXTRA_KELE_ARGS[@]:-}"
else
  $INHIBIT uv run python -m src.project.kele \
    --experiment "$EXPERIMENT" evaluate --output "$OUT_DIR" \
    "${EXTRA_KELE_ARGS[@]:-}"
fi

echo
echo "Eval finished: $(date)"

# ── Comparison ───────────────────────────────────────────────────────────────
if $DO_COMPARE; then
  echo
  echo "=== Comparison vs $BASELINE_DIR (gpt-4o-2024-11-20 baseline) ==="
  uv run python -m src.project.evaluate --compare "$BASELINE_DIR" "$OUT_DIR"

  # Per-stage state accuracy table (compare.py only prints overall)
  if command -v jq &>/dev/null; then
    echo
    echo "Per-stage state accuracy:"
    printf "  %-10s %15s %15s\n" "stage" "baseline" "$MODEL"
    printf "  %-10s %15s %15s\n" "----------" "---------------" "---------------"
    for stage in a b c d e overall; do
      if [[ "$stage" == "overall" ]]; then
        BASE=$(jq -r '.state_accuracy.overall' "$BASELINE_DIR/metrics_summary.json" 2>/dev/null || echo "?")
        THIS=$(jq -r '.state_accuracy.overall' "$OUT_DIR/metrics_summary.json"      2>/dev/null || echo "?")
      else
        BASE=$(jq -r ".state_accuracy.per_stage.$stage" "$BASELINE_DIR/metrics_summary.json" 2>/dev/null || echo "?")
        THIS=$(jq -r ".state_accuracy.per_stage.$stage" "$OUT_DIR/metrics_summary.json"      2>/dev/null || echo "?")
      fi
      printf "  %-10s %15s %15s\n" "$stage" "$BASE" "$THIS"
    done
  else
    echo "(install jq to see per-stage state accuracy comparison)"
  fi
fi

echo
echo "=== Done ==="
echo "Run log: $RUN_LOG"
echo "Results: $OUT_DIR/"
