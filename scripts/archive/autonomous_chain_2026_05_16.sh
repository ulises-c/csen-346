#!/usr/bin/env bash
# Autonomous chainer for the 2026-05-16 → 2026-05-17 weekend window.
#
# Waits for the in-flight Gemma 4 31B full run to complete, then runs as many
# follow-up experiments as fit before the Sun 15:00 PDT deadline. Threshold:
# 30 min minimum slack to start a new experiment (each takes ~50-60 min for
# n=50 Gemma, so the last one may finish slightly past 15:00 — acceptable).
#
# Follow-up queue (priority order, fills n=50 leaderboard gaps):
#   1. Gemma + 10-shot exemplars (LLM-only) — isolates Gemma vs BERT contribution
#   2. Gemma standalone (no exemplars, n=50) — strengthens the n=50 anchor
#   3. Gemma + 5-shot exemplars — sweep point for the few-shot curve
#
# Crash-safe: if Gemma full crashes, this chainer sleeps forever waiting for
# its metrics_summary.json — safer failure mode (manual intervention only).

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

GEMMA_METRICS="results/gemma4-31b-local-unified/metrics_summary.json"
DEADLINE_EPOCH=$(date -d "2026-05-17 15:00:00 PDT" +%s)
MIN_SLACK_SECONDS=1800   # 30 min minimum before starting a new experiment

LOG="logs/autonomous_chain_$(date -u +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Autonomous chainer v2 started: $(date) ==="
echo "Deadline: Sun 2026-05-17 15:00 PDT"
echo "Waiting for Gemma full to complete: $GEMMA_METRICS"
echo

# Poll for Gemma full completion (metrics_summary.json created at end of eval)
while [[ ! -f "$GEMMA_METRICS" ]]; do
  sleep 300
done

echo "Gemma full completed: $(date)"
echo "--- Gemma full metrics ---"
cat "$GEMMA_METRICS"
echo
echo "---"
echo

# Helper: print human-readable time remaining
time_remaining() {
  local now
  now=$(date +%s)
  local rem=$(( DEADLINE_EPOCH - now ))
  if [[ $rem -lt 0 ]]; then
    echo "0h 0m (deadline passed)"
  else
    echo "$(( rem / 3600 ))h $(( (rem % 3600) / 60 ))m"
  fi
}

# Helper: wait for port 8080 to be free
wait_for_port_free() {
  local tries=30
  while curl -s --max-time 2 http://localhost:8080/v1/models > /dev/null 2>&1; do
    echo "Port 8080 still serving — waiting..."
    sleep 30
    tries=$(( tries - 1 ))
    if [[ $tries -le 0 ]]; then
      echo "ERROR: port 8080 still busy after 15 min, aborting follow-up." >&2
      return 1
    fi
  done
  return 0
}

# Helper: run one follow-up experiment
# Args: $1 = output dir suffix, $2 = KELE_FEW_SHOT_N value (or "" for no exemplars)
run_followup() {
  local out_suffix="$1"
  local few_shot_n="$2"
  local out_dir="results/gemma4-31b-local-${out_suffix}"

  # Re-check time budget
  local now
  now=$(date +%s)
  local rem=$(( DEADLINE_EPOCH - now ))
  if [[ $rem -lt $MIN_SLACK_SECONDS ]]; then
    echo "Insufficient time ($(time_remaining)) for experiment '$out_suffix'. Stopping chain."
    return 1
  fi

  echo "=== Follow-up: $out_suffix  (time remaining: $(time_remaining)) ==="
  echo "Output: $out_dir"
  echo

  wait_for_port_free || return 1

  mkdir -p "$out_dir"
  TS=$(date -u +%Y%m%dT%H%M%S)

  # Boot server in background
  nohup bash "$ROOT/scripts/serve_gemma4_31b_q5.sh" > "$out_dir/server_${TS}.log" 2>&1 &
  local srv_pid=$!

  # Wait for server ready (up to 6 min cold load)
  echo -n "Booting Gemma 4 31B server (PID $srv_pid) "
  local ready=false
  for _ in $(seq 1 180); do
    if curl -s --max-time 3 http://localhost:8080/v1/models 2>/dev/null | grep -q '"Gemma 4 31B"'; then
      ready=true
      break
    fi
    if ! kill -0 "$srv_pid" 2>/dev/null; then
      echo " FAILED — server PID $srv_pid died."
      tail -20 "$out_dir/server_${TS}.log" >&2
      return 1
    fi
    echo -n "."
    sleep 2
  done
  if ! $ready; then
    echo " TIMEOUT after 360s"
    kill "$srv_pid" 2>/dev/null
    return 1
  fi
  echo " ready"

  # Run eval
  local env_prefix=""
  if [[ -n "$few_shot_n" ]]; then
    env_prefix="KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=$few_shot_n"
    echo "Few-shot mode: $few_shot_n exemplars"
  else
    echo "Few-shot mode: disabled (zero exemplars)"
  fi

  PATH="$ROOT/.venv/bin:$PATH" \
    bash -c "${env_prefix} uv run python -m src.project.kele \
      --experiment gemma4-31b-local \
      --unified \
      test --n 50 --output '$out_dir'"
  local eval_exit=$?

  # Teardown
  kill "$srv_pid" 2>/dev/null || true
  sleep 5
  kill -9 "$srv_pid" 2>/dev/null || true

  echo
  echo "=== $out_suffix complete (exit=$eval_exit) at $(date) ==="
  if [[ -f "$out_dir/metrics_summary.json" ]]; then
    cat "$out_dir/metrics_summary.json"
  fi
  echo
  return $eval_exit
}

# Defensive sleep — let Gemma full's server fully tear down
sleep 30

# Priority-ordered queue
run_followup "fewshot10-n50"    "10" || true
run_followup "no-exemplars-n50" ""   || true
run_followup "fewshot5-n50"     "5"  || true

echo "=== Chainer v2 done: $(date) ==="
echo "Final time remaining: $(time_remaining)"
