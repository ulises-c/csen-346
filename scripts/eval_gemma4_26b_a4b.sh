#!/usr/bin/env bash
# Orchestrator for the Gemma 4 26B-A4B local evaluation run.
# Single-server dual-role (teacher = consultant) on llama.cpp port 8080,
# mirroring the Qwen 27B / 35B-A3B pattern.
#
# Boots the llama.cpp server (if not already up), runs eval, compares against
# the gpt-4o-2024-11-20 baseline. Crash-safe via kele.py's existing per-item
# resume.
#
# Usage:
#   ./scripts/eval_gemma4_26b_a4b.sh smoke              # n=5
#   ./scripts/eval_gemma4_26b_a4b.sh mini               # n=25
#   ./scripts/eval_gemma4_26b_a4b.sh full               # n=681
#   ./scripts/eval_gemma4_26b_a4b.sh smoke --keep-server  # leave server running
#   ./scripts/eval_gemma4_26b_a4b.sh smoke --no-compare   # skip baseline comparison
#   ./scripts/eval_gemma4_26b_a4b.sh smoke --unified      # fusion architecture (single-call)
#
# Note: Gemma 4 has no thinking-mode equivalent (Qwen-only feature) — there is
# no --nothink flag. The --unified flag is supported and runs the same
# `SocraticTeachingSystemUnified` single-call architecture used for Qwen.
#
# Server policy: if a server is already serving "Gemma 4 26B A4B" on port 8080,
# reuse it and NEVER kill it (don't surprise other workflows). If we boot one,
# tear it down on exit unless --keep-server is passed.
#
# NOTE: shares port 8080 with the Qwen serve scripts — only one model fits at
# a time on the 5090's 32 GB VRAM. Stop any other server before invoking this.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ── Args ──────────────────────────────────────────────────────────────────────
MODE=""
KEEP_SERVER=false
DO_COMPARE=true
UNIFIED=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    smoke|mini|full) MODE="$1"; shift ;;
    --keep-server)   KEEP_SERVER=true; shift ;;
    --no-compare)    DO_COMPARE=false; shift ;;
    --unified)       UNIFIED=true; shift ;;
    --nothink)
      echo "Error: --nothink is not applicable to Gemma 4 (no thinking-mode equivalent)." >&2
      exit 1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 {smoke|mini|full} [--keep-server] [--no-compare] [--unified]" >&2
      exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Usage: $0 {smoke|mini|full} [--keep-server] [--no-compare] [--unified]" >&2
  exit 1
fi

# ── Mode config ───────────────────────────────────────────────────────────────
# Output dir suffix:
#   (default)    results/gemma4-26b-a4b-local-{smoke,mini}  /  results/gemma4-26b-a4b-local
#   --unified    -unified suffix appended
EXPERIMENT="gemma4-26b-a4b-local"
SUFFIX=""
if $UNIFIED; then
  SUFFIX="${SUFFIX}-unified"
fi
case "$MODE" in
  smoke) N=5;   OUT_DIR="results/gemma4-26b-a4b-local-smoke${SUFFIX}"; SUBCMD="test"     ;;
  mini)  N=25;  OUT_DIR="results/gemma4-26b-a4b-local-mini${SUFFIX}";  SUBCMD="test"     ;;
  full)  N=0;   OUT_DIR="results/gemma4-26b-a4b-local${SUFFIX}";       SUBCMD="evaluate" ;;
esac

# ── Constants / env-overridable paths ─────────────────────────────────────────
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/Documents/models/llama.cpp/build/bin/llama-server}"
GEMMA4_26B_WEIGHTS_DIR="${GEMMA4_26B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${GEMMA4_26B_WEIGHT_FILE:-gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf}"
EXPECTED_ALIAS="Gemma 4 26B A4B"
PORT="${PORT:-8080}"
LLAMA_URL="http://localhost:${PORT}"
CONFIG_FILE="configs/${EXPERIMENT}.env"
BASELINE_DIR="results/baseline"

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo "=== Gemma 4 26B-A4B Eval Orchestrator ==="
echo "Mode:      $MODE  (n=$N, subcmd=$SUBCMD)"
echo "Experiment:$EXPERIMENT"
echo "Output:    $OUT_DIR"
echo "Compare:   $DO_COMPARE  (vs $BASELINE_DIR)"
echo "Keep srv:  $KEEP_SERVER"
echo "Unified:   $UNIFIED"
echo "---"

preflight_check() {
  local name="$1" path="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: $name not found at $path" >&2
    exit 1
  fi
}

preflight_check "llama-server binary" "$LLAMA_SERVER"
preflight_check "weight file"         "$GEMMA4_26B_WEIGHTS_DIR/$WEIGHT_FILE"
preflight_check "config"              "$CONFIG_FILE"

if $DO_COMPARE && [[ ! -f "$BASELINE_DIR/metrics_summary.json" ]]; then
  echo "WARN: $BASELINE_DIR/metrics_summary.json missing — disabling comparison." >&2
  DO_COMPARE=false
fi

# GPU VRAM sanity (informational).
# Gemma 4 26B A4B Q5_K_XL at 220K ctx measured 30,688 MiB total / 2,080 MiB headroom.
# Headroom is exactly at the 2 GB rule, so warn if free < 30.5 GB.
if command -v nvidia-smi &>/dev/null; then
  VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  echo "GPU VRAM free: ${VRAM_FREE} MiB"
  if [[ "$VRAM_FREE" =~ ^[0-9]+$ ]] && [[ "$VRAM_FREE" -lt 30500 ]]; then
    echo "WARN: less than 30.5 GB VRAM free — server may OOM at 220K context." >&2
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
  echo "       Stop it first or override PORT=<other> $0 $MODE" >&2
  exit 1
else
  if probe_server >/dev/null; then
    echo "Existing server on $LLAMA_URL is loading the model — waiting for it..."
  else
    echo "Booting llama-server (cold load: 30-90 s for 20 GB GGUF)..."
    echo "Server log: $SERVER_LOG"
    bash "$ROOT/scripts/serve_gemma4_26b_a4b.sh" > "$SERVER_LOG" 2>&1 &
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
  INHIBIT="systemd-inhibit --what=sleep:idle --who=eval_gemma4_31b.sh --why=KELE-eval"
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
    printf "  %-10s %15s %15s\n" "stage" "baseline" "gemma4"
    printf "  %-10s %15s %15s\n" "----------" "---------------" "---------------"
    for stage in a b c d e overall; do
      if [[ "$stage" == "overall" ]]; then
        BASE=$(jq -r '.state_accuracy.overall' "$BASELINE_DIR/metrics_summary.json" 2>/dev/null || echo "?")
        GEMMA=$(jq -r '.state_accuracy.overall' "$OUT_DIR/metrics_summary.json"     2>/dev/null || echo "?")
      else
        BASE=$(jq -r ".state_accuracy.per_stage.$stage" "$BASELINE_DIR/metrics_summary.json" 2>/dev/null || echo "?")
        GEMMA=$(jq -r ".state_accuracy.per_stage.$stage" "$OUT_DIR/metrics_summary.json"     2>/dev/null || echo "?")
      fi
      printf "  %-10s %15s %15s\n" "$stage" "$BASE" "$GEMMA"
    done
  else
    echo "(install jq to see per-stage state accuracy comparison)"
  fi
fi

echo
echo "=== Done ==="
echo "Run log: $RUN_LOG"
echo "Results: $OUT_DIR/"
