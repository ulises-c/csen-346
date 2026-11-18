#!/usr/bin/env bash
# Run scripts/judge_validation.py on the R9700 for one or more local judge models.
# For each judge: start llama-server with reasoning off, run the validation on a
# finished run's dialogues, stop the server. Safe to restart (the validation
# resumes), time-capped, and the server is stopped on every exit path.
#
#   bash scripts/run_judge_validation_r9700.sh --preflight
#   bash scripts/run_judge_validation_r9700.sh            # about 30 minutes per judge
#
# Stop:     systemctl --user stop csen346-judgeval   (when started with systemd-run)
# Progress: cat ~/csen346-runs/STATUS
#
# Env: RUN_NAME (run directory under results/), JUDGES ("alias=path alias=path"),
#      CAP_S per judge (3600), LATEST_START_HOUR (18), PER_STAGE (30), REPEATS (2).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

RUN_NAME="${RUN_NAME:-r1-qwen35clf-gemma31b-fewshot10-n681-seed42}"
JUDGES="${JUDGES:-Qwen3.8-27B-Q5_K_M=$HOME/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_M.gguf Qwen3.6-27B-Q5_K_XL=$HOME/Documents/models/weights/Qwen3.6-27B-UD-Q5_K_XL.gguf}"
CAP_S="${CAP_S:-3600}"
# The GPU is loud and in a bedroom: do not start another judge late in the day.
LATEST_START_HOUR="${LATEST_START_HOUR:-18}"
PER_STAGE="${PER_STAGE:-30}"
REPEATS="${REPEATS:-2}"
PORT=8086
OUT="results/$RUN_NAME"
LOGS="$HOME/csen346-runs/judge-validation"
STATUS="$HOME/csen346-runs/STATUS"
mkdir -p "$LOGS"

SERVER_PID=""
JOB_PID=""
status() { echo "$(date -Is) judge-validation: $*" | tee -a "$STATUS"; }
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill "$SERVER_PID" 2>/dev/null
  for _ in 1 2 3 4 5 6; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
}
cleanup() {
  [[ -n "$JOB_PID" ]] && kill "$JOB_PID" 2>/dev/null
  stop_server
}
trap 'cleanup' EXIT
trap 'status "interrupted"; exit 130' INT TERM
vram_gb() { cat /sys/class/drm/card*/device/mem_info_vram_used | sort -n | tail -1 | awk '{printf "%.1f", $1/1e9}'; }

fail=0
check() {
  local what=$1; shift
  if "$@" >/dev/null 2>&1; then echo "  ok    $what"; else echo "  FAIL  $what"; fail=1; fi
}
echo "Preflight ($(hostname), $(git rev-parse --short HEAD), $(date -Is))"
check "dialogues present in $OUT" test -d "$OUT/dialogues"
for spec in $JUDGES; do check "judge GGUF ${spec%%=*}" test -f "${spec#*=}"; done
check "llama-server on PATH" command -v llama-server
check "uv environment" test -x .venv/bin/python
check "port $PORT free" bash -c "! ss -ltn | grep -q ':$PORT '"
check "GPU idle (VRAM in use: $(vram_gb) GB, limit 2)" awk -v v="$(vram_gb)" 'BEGIN { exit !(v < 2.0) }'
exec 9>"$HOME/csen346-runs/.lock"
check "no other GPU run in progress" flock -n 9
if [[ $fail -ne 0 ]]; then
  status "preflight FAILED, nothing started"
  exit 2
fi
if [[ "${1:-}" == "--preflight" ]]; then
  echo "Preflight passed. Nothing started."
  exit 0
fi

for spec in $JUDGES; do
  alias="${spec%%=*}"
  model="${spec#*=}"
  if (( 10#$(date +%H) > LATEST_START_HOUR )); then
    status "$alias skipped: past ${LATEST_START_HOUR}:59"
    continue
  fi
  LLAMA_ARG_REASONING=off llama-server -m "$model" -a "$alias" -ngl 99 -c 32768 -np 4 \
    --host 127.0.0.1 --port "$PORT" >"$LOGS/server_$alias.log" 2>&1 9>&- &
  SERVER_PID=$!
  ready=0
  for _ in $(seq 1 150); do
    sleep 2
    kill -0 "$SERVER_PID" 2>/dev/null || break
    curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { ready=1; break; }
  done
  if [[ $ready -ne 1 ]]; then
    status "$alias: server did not come up; see $LOGS/server_$alias.log"
    stop_server
    continue
  fi
  status "$alias: server ready (sha256 $(sha256sum "$model" | cut -c1-16)), VRAM $(vram_gb) GB; validation started"
  timeout "$CAP_S" .venv/bin/python scripts/judge_validation.py "$OUT" --backend api \
    --base-url "http://127.0.0.1:$PORT/v1" --api-key local --model "$alias" --workers 4 \
    --per-stage "$PER_STAGE" --repeats "$REPEATS" >"$LOGS/validation_$alias.log" 2>&1 &
  JOB_PID=$!
  wait "$JOB_PID"
  rc=$?
  JOB_PID=""
  stop_server
  status "$alias: validation exit $rc; $(grep -E '^Bad controls detected' "$LOGS/validation_$alias.log" | cut -c1-40)"
  sleep 5
done
status "finished; VRAM now $(vram_gb) GB"
